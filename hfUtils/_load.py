from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
def load_model_and_tokenizer(model_name = "google/gemma-2b-it"):
    
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=nf4_config,
                                                device_map="auto")
    # model.to(device)
    model.eval()
    
    return tokenizer, model