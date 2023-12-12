#!/bin/sh

# Define datasets, methods, and models to loop over
datasets=("citeseer" "cora" "pubmed")
methods=("baseline" "f_similarity" "s_similarity" "e_similarity" "fs_similarity" "fe_similarity" "se_similarity" "all")
models=("gcn")

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
