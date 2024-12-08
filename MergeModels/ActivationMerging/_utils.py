import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
from functools import partial
from ..utils.utils import set_random_seed, align_tokenizers_and_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
from copy import deepcopy


def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

@torch.no_grad()
def get_calib_feat(model, tokenizer):
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))

    device = model.device

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
        
    for k, v in input_dict.items():
        input_dict[k] = sum(v).float()
        
    return input_dict

@torch.no_grad()
def merge(finetuned_model_names, pretrained_model_name, topk=0.05, scaling=1.5, **kwargs):

    print('loading models...')
    models_to_merge, finetuned_tokenizers, finetuned_configs = [], [], []
    for finetuned_model_name in finetuned_model_names:
        try:
            finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, device_map='cpu', torch_dtype=torch.bfloat16)
            finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
            finetuned_config = AutoConfig.from_pretrained(finetuned_model_name)
            models_to_merge.append(finetuned_model)
            finetuned_tokenizers.append(finetuned_tokenizer)
            finetuned_configs.append(finetuned_config)
        except Exception as e:
            print(f"Model {finetuned_model_name} could not be loaded.")
            print(f"Reason: {e}")

    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, torch_dtype=torch.bfloat16)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
    pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)

    
    logger = logging.getLogger(__name__)
    # align the tokens of pretrained and finetuned tokenizer
    align_tokenizers_and_embeddings(pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer,
                                    pretrained_config=pretrained_config, finetuned_models=models_to_merge,
                                    finetuned_tokenizers=finetuned_tokenizers, finetuned_configs=finetuned_configs, logger=logger)
    
    print('getting calibration features...')
    scale_dicts = []
    for i in range(len(models_to_merge)):
        model = models_to_merge[i].to('cuda')
        tokenizer = finetuned_tokenizers[i]
        scale_dict = get_calib_feat(model, tokenizer)
        scale_dicts.append(scale_dict)
        model = model.to('cpu')
    
    pretrained_model = pretrained_model.to('cuda')
    pretrained_scale_dict = get_calib_feat(pretrained_model, pretrained_tokenizer)
    pretrained_model = pretrained_model.to('cpu')
    
    layer_mapping_dicts = []
    for i in range(len(models_to_merge)):
        layer_mapping_dict = {}
        for name, param in models_to_merge[i].named_modules():
            if isinstance(param, nn.Linear):
                layer_mapping_dict[name] = param
        layer_mapping_dicts.append(layer_mapping_dict)
    
    pretrained_layer_mapping_dict = {}
    for name, param in pretrained_model.named_modules():
        if isinstance(param, nn.Linear):
            pretrained_layer_mapping_dict[name] = param
    
    
    merged_model = deepcopy(pretrained_model)
    merged_layer_mapping_dict = {}
    for name, param in pretrained_model.named_modules():
        if isinstance(param, nn.Linear):
            merged_layer_mapping_dict[name] = param
    print('merging models...')
    # make a copy of the pretrained model
    for name, param in tqdm(pretrained_layer_mapping_dict.items(), total=len(pretrained_layer_mapping_dict)):
        base_importance = torch.softmax(pretrained_scale_dict[name],dim=0)
        base_importance = base_importance / base_importance.max()
        topk = torch.topk(base_importance, int(base_importance.numel() * topk)).indices
        important_clone = param.weight[:,topk].clone()
        for i in range(len(models_to_merge)):
            scale_dict = scale_dicts[i]
            layer_mapping_dict = layer_mapping_dicts[i]
            scale = torch.softmax(scale_dict[name],dim=0)
            scale = scale / scale.max()
            delta = layer_mapping_dicts[i][name].weight - pretrained_layer_mapping_dict[name].weight
            delta = delta * scale[None, :]
            merged_layer_mapping_dict[name].weight.data += delta*scaling
        merged_layer_mapping_dict[name].weight[:,topk].data = important_clone
    return merged_model, pretrained_tokenizer