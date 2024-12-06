import pandas as pd
import numpy as np
import os
from vllm import SamplingParams

problem_prompt = (
    "You will be given multiple choice questions about {subject}.\n\n Here are some examples:\n\n{examples}",
    "The following are multiple choice questions (with answers) about {subject}.\n\n{examples}",
    "Answer: {answer}",
    "{answer}"
)

def get_input_kwargs(tokenizer, choices = ["A", "B", "C", "D"]):
    
    ch = choices.copy()
    for i in range(len(choices)):
        ch[i] = problem_prompt[-1].format(answer=choices[i])
        
    force_words_ids = [tokenizer(ch, add_special_tokens=False).input_ids]
    max_new_tokens = 8
    num_beams = 4
    do_sample = False
    top_p = 1.0
    
    return {"force_words_ids": force_words_ids, "max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p}

def get_subject(data_dir):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    return subjects

def get_test_data_chat_template(data_path = './MMLU', choices = ["A", "B", "C", "D"], n_examples = None):
    subjects = get_subject(data_path)
    inputs = []
    Answers = []
    ptr = [0]
    for k,subject in enumerate(subjects):
        dev_df = pd.read_csv(os.path.join(data_path, "dev", "{}_dev.csv".format(subject)), header=None).to_numpy()
        test_df = pd.read_csv(os.path.join(data_path, "test", "{}_test.csv".format(subject)), header=None).to_numpy()
        
        n_choices = dev_df.shape[1] - 2
        
        subject_str = subject.replace('_',' ')
        examples = ""
        if n_examples is not None:
            n_ = min(n_examples, dev_df.shape[0])
        else:
            n_ = dev_df.shape[0]
        for i in range(n_):
            # prompt += dev_df[i][0]
            examples += dev_df[i][0]
            
            for j in range(n_choices):
                examples += f"\n{choices[j]}. {dev_df[i][j+1]}"

            examples += "\n"
            examples += problem_prompt[2].format(answer=dev_df[i][n_choices+1])
            examples += "\n\n"
            
        system_prompt = problem_prompt[0].format(subject=subject_str, examples=examples)
        
        for i in range(test_df.shape[0]):
            inputs.append([])
            inputs[-1].append({"role": "system", "content": system_prompt})
            prompt = test_df[i][0]
            
            for j in range(n_choices):
                prompt += f"\n{choices[j]}. {test_df[i][j+1]}"

            inputs[-1].append({"role": "user", "content": prompt})
            Answers.append(test_df[i][n_choices+1])
        
        ptr.append(ptr[-1]+test_df.shape[0])
        
    return inputs, Answers, subjects, ptr

def get_test_data_plain(data_path = './MMLU', choices = ["A", "B", "C", "D"], n_examples = None):
    subjects = get_subject(data_path)
    inputs = []
    Answers = []
    ptr = [0]
    for k,subject in enumerate(subjects):
        dev_df = pd.read_csv(os.path.join(data_path, "dev", "{}_dev.csv".format(subject)), header=None).to_numpy()
        test_df = pd.read_csv(os.path.join(data_path, "test", "{}_test.csv".format(subject)), header=None).to_numpy()
        
        n_choices = dev_df.shape[1] - 2
        
        subject_str = subject.replace('_',' ')
        examples = ""
        if n_examples is not None:
            n_ = min(n_examples, dev_df.shape[0])
        else:
            n_ = dev_df.shape[0]
        for i in range(n_):
            # prompt += dev_df[i][0]
            examples += dev_df[i][0]
            
            for j in range(n_choices):
                examples += f"\n{choices[j]}. {dev_df[i][j+1]}"

            examples += "\n"
            examples += problem_prompt[2].format(answer=dev_df[i][n_choices+1])
            examples += "\n\n"
            
        system_prompt = problem_prompt[1].format(subject=subject_str, examples=examples)
        
        for i in range(test_df.shape[0]):
            prompt = test_df[i][0]
            for j in range(n_choices):
                prompt += f"\n{choices[j]}. {test_df[i][j+1]}"
                
            prompt += f"\nAnswer: "

            inputs.append(system_prompt+prompt)
            Answers.append(test_df[i][n_choices+1])
        
        ptr.append(ptr[-1]+test_df.shape[0])
    
    return inputs, Answers, subjects, ptr

def run_mmlu_benchmark(model, prompt_type, data_dir = 'Data/', choices = "letter", n_examples = None):
    data_dir = os.path.join(data_dir, "MMLU")
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Data directory not found.")
    
    if choices == "letter":
        choices = ["A", "B", "C", "D"]
    elif choices == "number":
        choices = ["1", "2", "3", "4"]
    else:
        raise NotImplementedError('choices should be either "letter" or "number"')
    
    if prompt_type == 'chat':
        inputs, Answers, subjects, ptr = get_test_data_chat_template(data_dir, choices, n_examples)
        inputs = model.get_tokenizer().apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)
    elif prompt_type == 'plain':
        inputs, Answers, subjects, ptr = get_test_data_plain(data_dir, choices, n_examples)
    else:
        raise NotImplementedError('problem_prompt should be either "chat" or "plain"')
    
    allowed_token_ids = model.get_tokenizer()(choices,add_special_tokens=False, return_tensors='np').input_ids.flatten().tolist()
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=1, top_p=1.0, allowed_token_ids=allowed_token_ids)
    
    results = model.generate(inputs, sampling_params=sampling)
    results = extract_outputs(results)
    
    overall_acc = 0
    persubject_acc = {}
    for k,subject in enumerate(subjects):
        acc = 0
        for i in range(ptr[k], ptr[k+1]):
            if Answers[i] == results[i]:
                acc += 1
        persubject_acc[subject] = acc/len(Answers[ptr[k]:ptr[k+1]])
        overall_acc += acc
    
    overall_acc /= len(Answers)
    
    persubject_acc['overall'] = overall_acc
    
    return persubject_acc

def extract_outputs(outputs):
    return [output.outputs[0].text for output in outputs]