import pandas as pd
import numpy as np
import os

problem_prompt = (
    "You will be given multiple choice questions about {subject}.\n\n Here are some examples:\n\n{examples}Respond with a clear answer to the multiple choice question clearly stating A, B, C, or D.",
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

def load_mmlu_to_ram(data_path = './MMLU', choices = ["A", "B", "C", "D"], n_examples = None):
    
    subjects = get_subject(data_path)
    dev_data = []
    test_data = []
    train_prompts = []
    test_prompts = []

    for k,subject in enumerate(subjects):
        dev_df = pd.read_csv(os.path.join(data_path, "dev", "{}_dev.csv".format(subject)), header=None).to_numpy()
        test_df = pd.read_csv(os.path.join(data_path, "test", "{}_test.csv".format(subject)), header=None).to_numpy()
        
        n_choices = dev_df.shape[1] - 2
        prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_',' ')}.\n\n"
        
        if n_examples is not None:
            n_ = min(n_examples, dev_df.shape[0])
        else:
            n_ = dev_df.shape[0]
        for i in range(n_):
            prompt += dev_df[i][0]
            
            for j in range(n_choices):
                prompt += f"\n{choices[j]}. {dev_df[i][j+1]}"

            prompt += f"\nAnswer: {dev_df[i][n_choices+1]}\n\n"
        train_prompts.append(prompt)
        
        test_prompts.append([])
        for i in range(test_df.shape[0]):
            prompt = test_df[i][0]
            for j in range(n_choices):
                prompt += f"\n{choices[j]}. {test_df[i][j+1]}"

            prompt += f"\nAnswer: "
            test_prompts[-1].append([prompt, f"{test_df[i][n_choices+1]}"])
            
        
        dev_data.append(dev_df)
        test_data.append(test_df)
        subjects[k] = subject.replace('_',' ')
    
    return train_prompts, test_prompts, subjects

def get_test_data_chat_template(data_path = './MMLU', choices = ["A", "B", "C", "D"], n_examples = None):
    subjects = get_subject(data_path)
    inputs = []
    Answers = []
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
    
    return inputs, Answers

def get_test_data_plain(data_path = './MMLU', choices = ["A", "B", "C", "D"], n_examples = None):
    subjects = get_subject(data_path)
    inputs = []
    Answers = []
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
    
    return inputs, Answers

def extract_answers(responses):
    return [response.split("\\boxed{")[-1].split("}")[0] for response in responses]