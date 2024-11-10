import pandas as pd
import numpy as np
import os

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
            test_prompts[-1].append([prompt, f"{test_df[i][n_choices+1]}\n\n"])
            
        
        dev_data.append(dev_df)
        test_data.append(test_df)
        subjects[k] = subject.replace('_',' ')
    
    return train_prompts, test_prompts, subjects