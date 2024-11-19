import json

problem_prompt = (
    "You will be given instructions that describes a task. Write a response that appropriately completes the request. Think step by step. Decorate the final answer with \\boxed{}",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request. Think step by step. Decorate the final answer with \\boxed{}",
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)

def load_test_data(jsonl_path='./MATH/MATH_test.jsonl'):
    with open(jsonl_path, 'r') as f:
        test_data = [json.loads(line) for line in f.readlines()]
    return test_data

def get_test_data_chat_template(jsonl_path='./MATH/MATH_test.jsonl'):
    data = load_test_data(jsonl_path)
    
    inputs = []
    Answers = []
    
    for i in range(len(data)):
        instr = data[i]['instruction']
        inputs.append([])
        inputs[-1].append({"role": "system", "content": problem_prompt[0]})
        inputs[-1].append({"role": "user", "content": data[i]['instruction']})
        response = data[i]['output']
        
        # extract what is inside \boxed{}
        response = response.split("\\boxed{")
        response = response[1].split("}")
        response = response[0]
        Answers.append(response)

    return inputs, Answers

def get_test_data_plain(jsonl_path='./MATH/MATH_test.jsonl'):
    data = load_test_data(jsonl_path)
    
    inputs = []
    Answers = []
    
    for i in range(len(data)):
        instr = data[i]['instruction']
        
        prompt = problem_prompt[1] + '\n\n' + problem_prompt[2].format(instruction=instr)
        inputs.append(prompt)
        
        response = data[i]['output']
        response = response.split("\\boxed{")
        response = response[1].split("}")
        response = response[0]
        Answers.append(response)

    return inputs, Answers

def extract_answers(responses):
    return [response.split("\\boxed{")[-1].split("}")[0] for response in responses]