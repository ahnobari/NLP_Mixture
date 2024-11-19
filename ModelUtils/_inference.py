from tqdm.auto import trange, tqdm
import torch

def run_model(inputs, model, tokenizer, **kwargs):
    outputs = []
    for i in trange(len(inputs)):
        input_tensor = torch.tensor(inputs[i]).unsqueeze(0).to(model.device)
        output = model.generate(input_tensor, **kwargs)[0][len(inputs[i]):]
        output = tokenizer.decode(output, skip_special_tokens=True)
        outputs.append(output)
    
    return outputs

def run_model_batch(inputs, attention_masks, model, tokenizer, batch_size=3, **kwargs):
    outputs = []
    for i in trange(0, len(inputs), batch_size):
        input_tensor = torch.tensor(inputs[i:i+batch_size]).to(model.device)
        attention_mask = torch.tensor(attention_masks[i:i+batch_size]).to(model.device)
        output = model.generate(input_ids = input_tensor, attention_mask=attention_mask, pad_token_id = tokenizer.eos_token_id, **kwargs)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        outputs += output
    return outputs
    