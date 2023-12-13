#!/bin/sh

# Define datasets, methods, and models to loop over
datasets=("cora" "pubmed")
methods=("all" "se_similarity" "fe_similarity" "fs_similarity" "e_similarity" "s_similarity" "f_similarity" "baseline")
models=("simple_gcn")

# Loop over datasets
for dataset in "${datasets[@]}"
do
    echo "Dataset: $dataset"
    for method in "${methods[@]}"
    do
        echo "Method: $method"
        for model in "${models[@]}"
        do
            echo "Model: $model"
                python train_entropy_density_graphcentral_ts.py --dataset "$dataset" --method "$method" --model "$model"
            done
        done
    done
done
