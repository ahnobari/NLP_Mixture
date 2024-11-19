from tqdm.auto import tqdm, trange

def tokenize_chat(inputs, tokenizer):
    for i in trange(len(inputs)):
        if i == 0:
            tokenized_inputs = [tokenizer.apply_chat_template(inputs[i],add_generation_prompt=True)]
        else:
            tokenized_inputs += [tokenizer.apply_chat_template(inputs[i],add_generation_prompt=True)]
    
    return tokenized_inputs

def tokenize_plain(inputs, tokenizer):
    for i in trange(len(inputs)):
        if i == 0:
            tokenized_inputs = [tokenizer(inputs[i])['input_ids']]
        else:
            tokenized_inputs += [tokenizer(inputs[i])['input_ids']]

    return tokenized_inputs