from config import *
from helpers import load_randomalpdata, create_mask, preprocess_features, preprocess_adj


# Define hyperparameters based on the dataset and method
if args.dataset == "citeseer":
    hyperparameterC = 0.85 if args.method == "AGE+FDS+SDS" else 0.9
elif args.dataset == "cora":
    hyperparameterC = 0.95 if args.method == "AGE+FDS+SDS" else 0.99
elif args.dataset == "pubmed":
    hyperparameterC = 0.995
else:
    hyperparameterC = 0

assert hyperparameterC, "Hyperparameter not set"
assert args.model in {
    "baseline",
    "AGE+FDS",
    "AGE+SDS",
    "AGE+EDS",
    "AGE+FDS+SDS",
}, "Incorrect method"

MAC = []
MIC = []
for idx in range(10):
    (
        adj,
        features,
        y_train,
        y_val,
        y_test,
        train_mask,
        val_mask,
        test_mask,
        idx_train,
        labels,
        graph,
    ) = load_randomalpdata(args.dataset, idx, args.inicount)

    message_passing = adj * adj
    raw_features = features.todense()
    features = preprocess_features(features)
    support = []

