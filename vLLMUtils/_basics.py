from vllm import LLM
import torch

def load_model(model_name, **kwargs):
    
    gpu_count = torch.cuda.device_count()
    kwargs['tensor_parallel_size'] = gpu_count
    kwargs['trust_remote_code'] = True
    kwargs['max_model_len'] = 39808 ## This is 4090 Specific Limit
    
    model = LLM(model=model_name, **kwargs)
    
    return model

def extract_outputs(outputs):
    return [output.outputs[0].text for output in outputs]