from helpers import *
from gcn import GCN

import numpy as np
import sklearn
import tensorflow as tf

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
assert args.method in {
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
    ) = load_randomalpdata(args.dataset, idx, int(args.inicount))
    message_passing = adj * adj
    raw_features = features.todense()
    # print(type(features))
    features = preprocess_features(features)
    # tuple_features = sparse_to_tuple(sparse_features)


    # print(features[1])
    # exit(0)
    support = [preprocess_adj(adj)]
    num_supports = 1

    num_classes = args.num_classes
    num_labels = num_classes * 20

    sparse_features = tf.sparse.SparseTensor(indices=features[0], values=features[1], dense_shape=features[2])
    model = GCN(input_dim=features[2][1], output_dim=y_train.shape[1], num_nonzero_features=features[1].shape)

    cost_val = []

    normcen = np.loadtxt("res/" + args.dataset + "/graphcentrality/normcen")
    cenperc = np.asarray([perc(normcen, i) for i in range(len(normcen))])

    for epoch in range(args.epochs):
        gamma = np.random.beta(1, 1.005 - hyperparameterC ** epoch)
        if args.method == 'baseline':
            alpha = beta = delta = epsilon = (1 - gamma) / 2
        elif args.method == 'AGE+FDS+SDS':
            alpha = beta = delta = epsilon = (1 - gamma) / 4
        else:
            alpha = beta = delta = epsilon = (1 - gamma) / 3
        outs = model((sparse_features, y_train, adj), train_mask)
        print(outs[3])
        if len(idx_train) < num_labels:
            curr_features = raw_features[idx_train, :]
            curr_features = sklearn.preprocessing.normalize(curr_features)
            raw_features = sklearn.preprocessing.normalize(raw_features)
