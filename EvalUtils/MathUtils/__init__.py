from .python_executor import PythonExecutor
from .data_loader import load_data
from .parser import parse_question, parse_ground_truth, run_execute
from .trajectory import extract_program
from .evaluate import evaluate as eval
from tqdm.auto import tqdm
import os
from vllm import SamplingParams

PATH = __file__

def load_prompt(data_name, prompt_type):
    if data_name in ['gsm_hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']:
        data_name = "gsm8k"
    if data_name in ['math_oai', "hungarian_exam"]:
        data_name = "math"
    if data_name in ['sat_math']:
        data_name = "mmlu_stem"
    if prompt_type in ['platypus_fs']:
        prompt_type = "cot"
    if prompt_type in ['tool-integrated']:
        prompt_type = "tora"

    if prompt_type in ['cot', 'pal', 'tora']:
        prompt_path = os.path.join(os.path.dirname(PATH), f"prompts/{prompt_type}/{data_name}.md")
        if not os.path.exists(prompt_path):
            prompt_path = "./prompts/{}.md".format(prompt_type)
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                prompt = fp.read().strip() + "\n\n\n"
        else:
            print(f"Error: prompt file {prompt_path} not found")
            prompt = ""
    else:
        prompt = ""
    return prompt

def construct_prompt(example, data_name, prompt_type):
    # Base models
    if prompt_type in ["direct", "cot", "pal", "tool-integrated"]:
        demo_prompt = load_prompt(data_name, prompt_type)
        if prompt_type in ["direct", "cot"]:
            if data_name in ["minerva_math", "math", "math_oai", "mmlu_stem", "sat_math", "mathqa", "hungarian_exam"]:
                context = f"Problem:\n{example['question']}\nSolution:"
            else:
                context = f"Question: {example['question']}\nAnswer:"
            full_prompt = demo_prompt + context
        elif prompt_type == "pal":
            context = f"Question: {example['question']}"
            full_prompt = demo_prompt + context
        elif prompt_type in ['tool-integreted']:
            context = f"Question: {example['question']}\n\nSolution:"
            full_prompt = demo_prompt + context

    # SFT models
    elif prompt_type in ['self-instruct', 'tora']:
        full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
    elif prompt_type in ['self-instruct-boxed']:
        full_prompt = f"<|user|>\n{example['question']}\nEnclose the final answer using \\boxed{{}}.\n<|assistant|>\n"
    elif prompt_type == "wizard_zs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif prompt_type == "deepseek-math":
        full_prompt = (
            "User: {instruction}\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}.\n\nAssistant:"
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif prompt_type == "kpmath":
        full_prompt = (
            'User: Please reason step by step and put your final answer at the end '
            'with "The answer is: ".\n\n{instruction}\n\nAssistant:'
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    else:
        raise NotImplementedError(prompt_type)
    return full_prompt

def run_math_benchmark(LLM, data_set_name, prompt_type, split="test", data_dir="./Data", n_sampling=1, temperature=0.0, top_p=1.0, max_tokens_per_call=1024):
    
    # init python executor
    if "pal" in prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
        
    examples = load_data(data_set_name, split, data_dir)
    
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, data_set_name)
        gt_cot, gt_ans = parse_ground_truth(example, data_set_name)
        full_prompt = construct_prompt(example, data_set_name, prompt_type)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)
    
    input_prompts = [sample['prompt'] for sample in samples for _ in range(n_sampling)]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if prompt_type in ['cot', 'pal'] else 4

    # stop words TODO: make it more general
    stop_words = ["</s>"]

    if prompt_type in ['cot']:
        stop_words.extend(["\n\nQuestion:", "\n\nProblem:"])
    if prompt_type in ['pal', 'tool-integrated', 'tora']:
        stop_words.extend(["\n\n---", "```output"])
    elif prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_words.extend(["Instruction", "Response"])

    # start inference
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
       
        
        outputs = LLM.generate(prompts, SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens_per_call,
                        n=1,
                        stop=stop_words,
                    ))

        outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]
        
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif prompt_type == "cot":
                end_prompts.append((i, query))
            elif ("boxed" not in output and output.endswith("```")):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    
    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        codes.append(code)

    # extract preds
    results = [run_execute(executor, code, prompt_type, data_set_name) for code in codes]

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*n_sampling: (i+1)*n_sampling]
        result = results[i*n_sampling: (i+1)*n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples, result_json = eval(samples=all_samples, data_name=data_set_name, prompt_type=prompt_type, execute=True)

    return result_json