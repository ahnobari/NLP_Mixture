from tqdm.auto import tqdm, trange
import torch

def tokenize_chat(inputs, tokenizer):
    for i in trange(len(inputs)):
        if i == 0:
            tokenized_inputs = [tokenizer.apply_chat_template(inputs[i],add_generation_prompt=True)]
        else:
            tokenized_inputs += [tokenizer.apply_chat_template(inputs[i],add_generation_prompt=True)]
    
    return tokenized_inputs

def tokenize_chat_batch(inputs, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer.apply_chat_template(inputs, padding=True, add_generation_prompt=True, return_tensors="pt")
    n_padding = torch.sum((tokens == tokenizer.pad_token_id)[:,:-1] * (tokens == tokenizer.pad_token_id)[:,1:],dim=1)
    n_padding[n_padding != 0] += 1
    attention_mask = 1-torch.cumsum(torch.nn.functional.one_hot(tokens.shape[1]-n_padding, num_classes=tokens.shape[1]+1),dim=1)
    attention_mask = attention_mask[:,:-1]
    
    return tokens, attention_mask

def tokenize_plain(inputs, tokenizer):
    for i in trange(len(inputs)):
        if i == 0:
            tokenized_inputs = [tokenizer(inputs[i])['input_ids']]
        else:
            tokenized_inputs += [tokenizer(inputs[i])['input_ids']]

    return tokenized_inputs