from evalplus.data import get_human_eval_plus, get_mbpp_plus
from tqdm.auto import tqdm
from ._CodeTools import evaluate_minimal
from vllm import SamplingParams
from evalplus.codegen import sanitize

def run_code_test(model, dataset):
    if dataset == "humaneval":
        data = get_human_eval_plus()
    elif dataset == "mbpp":
        data = get_mbpp_plus()
    else:
        raise ValueError(f"Invalid dataset {dataset}. Must be humaneval or mbpp.")
    
    prompts = []
    entry_points = []
    task_ids = []
    for task_id, task in tqdm(data.items()):
        # prompt = task['prompt']
        prompts.append(task['prompt'])
        entry_points.append(task['entry_point'])
        task_ids.append(task_id)
    
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=1024, top_p=1.0)
    
    results = model.generate(prompts, sampling_params=sampling)
    results = extract_outputs(results)
    
    sanitized_results = []

    for result, entry_point in tqdm(zip(results, entry_points), total=len(results)):
        sanitized_result = sanitize(result, entry_point)
        sanitized_results.append(sanitized_result)
        
    samples = []
    for task_id, result in zip(task_ids, sanitized_results):
        samples.append({"task_id": task_id, "solution": result})
        
    final_acc = evaluate_minimal(dataset, samples)
    
    return final_acc

def extract_outputs(outputs):
    return [output.outputs[0].text.replace("\t", "    ") for output in outputs]