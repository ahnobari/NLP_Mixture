#!/bin/bash

models=('flitered_base/average_merging'
        'flitered_base/mask_merging'
        'flitered_base/task_arithmetic'
        'flitered_base/ties_merging'
        'flitered_base/widen_merging'
        'flitered_instruct/average_merging'
        'flitered_instruct/mask_merging'
        'flitered_instruct/task_arithmetic'
        'flitered_instruct/ties_merging'
        'flitered_instruct/widen_merging'
        'ActiveM_instruct/ActiveM_0.1_1.0'
        'ActiveM_instruct/ActiveM_0.01_1.0'
        'ActiveM_instruct/ActiveM_0.05_1.0'
        'ActiveM_instruct/ActiveM_0.1_2.0'
        'ActiveM_instruct/ActiveM_0.01_2.0'
        'ActiveM_instruct/ActiveM_0.05_2.0'
        'ActiveM_instruct/ActiveM_0.1_5.0'
        'ActiveM_instruct/ActiveM_0.01_5.0'
        'ActiveM_instruct/ActiveM_0.05_5.0'
        'ActiveM_base/ActiveM_0.1_1.0'
        'ActiveM_base/ActiveM_0.01_1.0'
        'ActiveM_base/ActiveM_0.05_1.0'
        'ActiveM_base/ActiveM_0.1_2.0'
        'ActiveM_base/ActiveM_0.01_2.0'
        'ActiveM_base/ActiveM_0.05_2.0'
        'ActiveM_base/ActiveM_0.1_5.0'
        'ActiveM_base/ActiveM_0.01_5.0'
        'ActiveM_base/ActiveM_0.05_5.0')

for model in "${models[@]}"
do
    echo "Running benchmark for $model"
    python run_benchmark.py --model "$model"
done