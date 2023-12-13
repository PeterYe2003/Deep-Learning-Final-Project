import pickle as pkl
import numpy as np
import networkx as nx
import random
import scipy.sparse as sp


def create_mask(idx, n):
    """Create mask."""
    mask = np.zeros(n)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_randomalpdata(dataset_str, iter, inicount):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    data = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            data.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(data)
    test_idx = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        test_idx_range = range(min(test_idx), max(test_idx) + 1)

        tx_ext = sp.lil_matrix((len(test_idx_range), x.shape[1]))
        tx_ext[sorted_test_idx - min(sorted_test_idx), :] = tx
        tx = tx_ext

        ty_ext = np.zeros((len(test_idx_range), y.shape[1]))
        ty_ext[sorted_test_idx - min(sorted_test_idx), :] = ty
        ty = ty_ext

        num_train_and_val = 2312
        num_classes = 6
    elif dataset_str == 'cora':
        num_train_and_val = 1708
        num_classes = 7
    else:
        num_train_and_val = 18717
        num_classes = 3

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[sorted_test_idx, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[sorted_test_idx, :]

    idx_val = [int(item) for item in open("source/" + dataset_str + "/val_idx" + str(iter) + ".txt").readlines()]
    train_candidates = list(set(range(0, num_train_and_val)) - set(idx_val))  # train candidate, not test not valid
    train_and_val_labels = labels[train_candidates]
    gt_labels = np.argmax(train_and_val_labels, axis=1)
    idx_train = []
    for i in range(num_classes):
        idx = np.where(gt_labels == i)
        rand_idx = random.sample(range(0, idx[0].shape[0]), inicount)
        idx_train += list(np.asarray(train_candidates)[list(idx[0][rand_idx])])

    train_mask = create_mask(idx_train, labels.shape[0])
    val_mask = create_mask(idx_val, labels.shape[0])
    test_mask = create_mask(sorted_test_idx, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        return coords, mx.data, mx.shape

    if isinstance(sparse_mx, list):
        sparse_mx = [to_tuple(sparse_mx[i]) for i in range(len(sparse_mx))]
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    deg = np.array(features.sum(axis=1))
    deg_inv = np.power(deg, -1).flatten()
    deg_inv[np.isinf(deg_inv)] = 0
    D_inv = sp.diags(deg_inv)
    features = D_inv.dot(features)
    return sparse_to_tuple(features)
    # return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    if not sp.isspmatrix_coo(adj):
        adj = sp.coo_matrix(adj)
    deg_inv = np.array(adj.sum(axis=1))
    deg_inv_rt = np.power(deg_inv, -0.5).flatten()
    deg_inv_rt[np.isinf(deg_inv_rt)] = 0
    D_inv_rt = sp.diags(deg_inv_rt)
    return adj.dot(D_inv_rt).transpose().dot(D_inv_rt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


# def construct_feed_dict(features, support, labels, labels_mask, placeholders):
#     """Construct feed dictionary."""
#     feed_dict = dict()
#     feed_dict.update({placeholders['labels']: labels})
#     feed_dict.update({placeholders['labels_mask']: labels_mask})
#     feed_dict.update({placeholders['features']: features})
#     feed_dict.update({placeholders['support']: support})
#     feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
#     return feed_dict

def perc(input_list, k):
        count = sum(1 for i in input_list if i < input_list[k])
        return count / len(input_list)

def percd(input_list, k):
        count = sum(1 for i in input_list if i > input_list[k])
        return count / len(input_list)
