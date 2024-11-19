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
