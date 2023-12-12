#!/bin/sh

# Define datasets, methods, and models to loop over
datasets=("citeseer" "cora" "pubmed")
methods=("fs_similarity" "fe_similarity" "se_similarity" "all")
models=("gcn" "simple_gcn")

# Define input parameters
start=0
end=200  # Modify this value to set the number of iterations
increment=1

# Loop over datasets
for dataset in "${datasets[@]}"
do
    echo "Dataset: $dataset"

    # Loop over methods
    for method in "${methods[@]}"
    do
        echo "Method: $method"

        # Loop over models
        for model in "${models[@]}"
        do
            echo "Model: $model"

            # Loop over the specified range for the 'i' variable
            for ((i = $start; i < $end; i += $increment))
            do
                python train_entropy_density_graphcentral_ts.py --dataset "$dataset" --method "$method" --model "$model"
                echo "Iteration: $i"
            done
        done
    done
done
