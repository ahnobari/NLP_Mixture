# %%
import warnings
warnings.filterwarnings("ignore")

import DataUtils.MATH as MATH
import DataUtils.MMLU as MMLU
from hfUtils import *
from ModelUtils import *

import numpy as np

import torch

# %%
tokenizer, model = load_model_and_tokenizer('meta-llama/Llama-3.2-1B-Instruct')

# %%
MMLU_inputs_chat, MMLU_Answers_chat = MMLU.get_test_data_chat_template()
MATH_inputs_chat, MATH_Answers_chat = MATH.get_test_data_chat_template()

MMLU_inputs_plain, MMLU_Answers_plain = MMLU.get_test_data_plain()
MATH_inputs_plain, MATH_Answers_plain = MATH.get_test_data_plain()

# This will take a few seconds
tokenized_MMLU_inputs_chat = tokenize_chat(MMLU_inputs_chat, tokenizer)
tokenized_MMLU_inputs_plain = tokenize_plain(MMLU_inputs_plain, tokenizer)

tokenized_MATH_inputs_chat = tokenize_chat(MATH_inputs_chat, tokenizer)
tokenized_MATH_inputs_plain = tokenize_plain(MATH_inputs_plain, tokenizer)

# %%
# print some sample tokenized inputs
rnd_idx = np.random.randint(0, len(MATH_inputs_chat))
print("MATH Chat Template Input: \n", tokenizer.decode(token_ids = tokenized_MATH_inputs_chat[rnd_idx]))
print("MATH Plain Input: \n", tokenizer.decode(token_ids = tokenized_MATH_inputs_plain[rnd_idx]))

# %%
MATH.get_input_kwargs(tokenizer)

# %%
results = run_model(tokenized_MATH_inputs_chat, model, tokenizer, **MATH.get_input_kwargs(tokenizer))

# %%
import pickle

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)


