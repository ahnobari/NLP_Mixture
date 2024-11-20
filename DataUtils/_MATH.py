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
        #response = response.split("\\boxed{")
        #response = response[1][:-1] #I'm assuming } is the last char, we need to fix this later.
        #response = response[0]
        response = extract_answer(response)
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
        #response = response.split("\\boxed{")
        #response = response[1] #I'm assuming } is the last char, we need to fix this later.
        #response = response[0]
        response = extract_answer(response)
        Answers.append(response)

    return inputs, Answers

def extract_answers(responses):
    return [extract_answer(response) for response in responses]


def get_input_kwargs(tokenizer, choices = ["A", "B", "C", "D"]):
    
    for i in range(len(choices)):
        choices[i] = problem_prompt[-1].format(answer=choices[i])
        
    force_words_ids = [tokenizer(choices, add_special_tokens=False).input_ids]
    max_new_tokens = 500
    num_beams = 2
    do_sample = False
    top_p = 0.9
    
    return {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p}

def find_first_unopened_closing_brace(a):
    x = 0
    for i, char in enumerate(a):
        if char == '{':
            x += 1 
        elif char == '}':
            x -= 1
        if x == -1:
            return a[:i]
        
    return a

def extract_answer(response):
    a = response.split("\\boxed{")[-1]
    a = find_first_unopened_closing_brace(a)
    return a
