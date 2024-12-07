import argparse
from EvalUtils import *
import uuid
import os
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path. Default: meta-llama/Llama-3.2-1B-Instruct")
argparser.add_argument("--mmlu_prompt_type", type=str, default="plain", help="Prompt type for MMLU benchmark. Default: plain. one of [chat, plain]")
argparser.add_argument("--mmlu_choices", type=str, default="letter", help="Choices for MMLU benchmark. Default: letter. one of [letter, number]")
argparser.add_argument("--data_dir", type=str, default="Data", help="Data directory. Default: Data")
argparser.add_argument("--mmlu_n_examples", type=int, default=None, help="Number of examples to use for MMLU. Default: None(All)")
argparser.add_argument("--math_benchmarks", type=str, default="math,gsm8k,math_oai,gsm_hard", help="Math benchmarks directory. Default: Data")
argparser.add_argument("--math_prompt_type", type=str, default="tora", help="Prompt type for Math benchmarks. Default: tora")
argparser.add_argument("--math_max_tokens", type=int, default=1024, help="Max tokens for Math benchmarks. Default: 1024")
argparser.add_argument("--save_path", type=str, default="results", help="Save path for results. Default: results, File name will be automatically generated")

args = argparser.parse_args()

# check if results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

print("loading model ...")
# Load model
model = load_model(args.model)

# Run MMLU benchmark
print("running MMLU benchmark ...")
mmlu_results = run_mmlu_benchmark(model, args.mmlu_prompt_type, args.data_dir, args.mmlu_choices, args.mmlu_n_examples)

print(f"MMLU accuracy: {mmlu_results['overall']:.2f}")

math_benchmarks = args.math_benchmarks.split(",")
math_results = {}

for benchmark in math_benchmarks:
    print(f"running {benchmark} benchmark ...")
    math_results[benchmark] = run_math_benchmark(model, benchmark, args.math_prompt_type, data_dir=args.data_dir, max_tokens_per_call=args.math_max_tokens)

# save results
print("saving results ...")

while True:
    model_safe = args.model.split("/")[-1].replace("-", "_").replace('.', '_')
    save_path = os.path.join(args.save_path, f"{model_safe}_{uuid.uuid4()}.json")
    if not os.path.exists(save_path):
        break
math_results["mmlu"] = mmlu_results

with open(save_path, "w") as f:
    json.dump(math_results, f)