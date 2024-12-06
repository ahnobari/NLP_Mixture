import argparse
from EvalUtils import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path. Default: meta-llama/Llama-3.2-1B-Instruct")
argparser.add_argument("--mmlu_prompt_type", type=str, default="plain", help="Prompt type for MMLU benchmark. Default: plain. one of [chat, plain]")
argparser.add_argument("--mmlu_choices", type=str, default="letter", help="Choices for MMLU benchmark. Default: letter. one of [letter, number]")
argparser.add_argument("--data_dir", type=str, default="Data", help="Data directory. Default: Data")
argparser.add_argument("--mmlu_n_examples", type=int, default=None, help="Number of examples to use for MMLU. Default: None(All)")