#!/bin/bash

models=('merged_models/task_arithmetic')

for model in "${models[@]}"
do
    echo "Running benchmark for $model"
    python run_benchmark.py --model "$model"
done