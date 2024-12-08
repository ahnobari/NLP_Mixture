from vllm import LLM
import torch
import os

def load_model(model_name_or_path, **kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false"
    )
    gpu_count = torch.cuda.device_count()
    kwargs['tensor_parallel_size'] = gpu_count
    kwargs['trust_remote_code'] = True
    kwargs['max_model_len'] = 30000 ## This is 4090 Specific Limit
    
    model = LLM(model=model_name_or_path, **kwargs)
    
    return model