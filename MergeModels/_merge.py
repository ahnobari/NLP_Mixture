import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .model_merging_methods.merging_methods import MergingMethod
from .utils.utils import set_random_seed, align_tokenizers_and_embeddings
from .utils.load_config import cache_dir
from copy import deepcopy
import torch

DEFAULTS = {
#    "merging_method_name": "average_merging",
    "scaling_coefficient": 1.0,
    "slerp_t": 0.5,
    "dot_threshold": 0.9995,
    "param_density": 0.9,
    "param_value_mask_rate": 0.8,
    "weight_format": "delta_weight",
    "weight_mask_rate": 0.1,
    "use_weight_rescale": False,
    "mask_strategy": "random",
    "mask_apply_method": "average_merging",
    "above_average_value_ratio": 1.0,
    "score_calibration_value": 1.0
}

def merge_models(pretrained_model_name: str, models_to_merge: list, finetuned_tokenizers: list, finetuned_configs: list, logger: logging.Logger, merging_method: MergingMethod, **kwargs):
    """
    merge models by merging method with merging_method_name and save them
    :param args: ArgumentParser, input argument parser
    :param models_to_merge: list, individual models that need to be merged
    :param finetuned_tokenizers: list of finetuned tokenizers
    :param finetuned_configs: list of finetuned configs
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method÷
    :return:
    """
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name, torch_dtype=torch.bfloat16)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
    pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)

    # align the tokens of pretrained and finetuned tokenizer
    align_tokenizers_and_embeddings(pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer,
                                    pretrained_config=pretrained_config, finetuned_models=models_to_merge,
                                    finetuned_tokenizers=finetuned_tokenizers, finetuned_configs=finetuned_configs, logger=logger)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # merged_model = merging_method.get_merged_model(merged_model=pretrained_model,
    #                                                models_to_merge=models_to_merge,
    #                                                exclude_param_names_regex=[],
    #                                                scaling_coefficient=kwargs['scaling_coefficient'],
    #                                                slerp_t=kwargs['slerp_t'],
    #                                                dot_threshold=kwargs['dot_threshold'],
    #                                                param_density=kwargs['param_density'],
    #                                                param_value_mask_rate=kwargs['param_value_mask_rate'],
    #                                                weight_format=kwargs['weight_format'],
    #                                                weight_mask_rates=kwargs['weight_mask_rates'],
    #                                                use_weight_rescale=kwargs['use_weight_rescale'],
    #                                                mask_strategy=kwargs['mask_strategy'],
    #                                                mask_apply_method=kwargs['mask_apply_method'],
    #                                                above_average_value_ratio=kwargs['above_average_value_ratio'],
    #                                                score_calibration_value=kwargs['score_calibration_value'])

    merged_model = merging_method.get_merged_model(merged_model=pretrained_model,
                                                   models_to_merge=models_to_merge,
                                                    exclude_param_names_regex=[], **kwargs)
    

    #save_model_name = f"{merging_method}_{'_'.join(models_to_merge)}_{pretrained_model_name}"
    #save_model_path = f"./merged_llms/{pretrained_model_name}/{'_'.join(models_to_merge)}/{save_model_name}"
    #logger.info(f"Saving merged models at {save_model_path}...")
    #merged_model.save_pretrained(save_directory=save_model_path)
    #pretrained_tokenizer.save_pretrained(save_directory=save_model_path)
    
    #logger.info(f"Merging of {'_'.join(models_to_merge)} with method {merging_method} is completed.")

    return merged_model, pretrained_tokenizer

def merge(finetuned_model_names, pretrained_model_name, merging_method_name, **kwargs):

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

    args = deepcopy(DEFAULTS)

    # default logger
    logger = logging.getLogger(__name__)

    if 'weight_mask_rates' not in kwargs:
        args['weight_mask_rates'] = [args['weight_mask_rate'] for _ in range(len(finetuned_model_names))]

    merging_method = MergingMethod(merging_method_name=merging_method_name)

    args['merging_method_name'] = merging_method_name

    merged = merge_models(pretrained_model_name, models_to_merge=models_to_merge, finetuned_tokenizers=finetuned_tokenizers,
                        finetuned_configs=finetuned_configs, logger=logger, merging_method=merging_method, **args, **kwargs)

    return merged