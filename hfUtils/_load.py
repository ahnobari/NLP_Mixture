from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_model_and_tokenizer(model_name = "google/gemma-2b-it"):
    
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype = torch.bfloat16,
                                                load_in_8bit=False,
                                                trust_remote_code=True)
    model.to(device)
    model.eval()
    
    return tokenizer, model