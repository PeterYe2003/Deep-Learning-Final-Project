import pickle as pkl
import numpy as np
import networkx as nx
import random
import scipy.sparse as sp


def sample_mask(idx, n):
    """Create mask."""
    mask = np.zeros(n)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_randomalpdata(dataset_str, iter, inicount):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        NL = 2312
        NC = 6
    elif dataset_str == 'cora':
        NL = 1708
        NC = 7
    else:
        NL = 18717
        NC = 3

    print(allx.shape)
    print(tx.shape)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print(features.shape)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # fixed 500 for validation read from file, choose random inicount per class as initial of al from the others for train
    '''
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    '''

    # size of idx_train, train_mask and y_train will incrementally increase
    idx_test = test_idx_range.tolist()
    idx_val = [int(item) for item in open("source/" + dataset_str + "/val_idx" + str(iter) + ".txt").readlines()]
    idx_traincand = list(set(range(0, NL)) - set(idx_val))  # train candiate, not test not valid
    nontestlabels = labels[idx_traincand]
    gtlabels = np.argmax(nontestlabels, axis=1)
    idx_train = []
    for i in range(NC):
        nodeidx = np.where(gtlabels == i)
        ridx = random.sample(range(0, nodeidx[0].shape[0]), inicount)
        idx_train += list(np.asarray(idx_traincand)[list(nodeidx[0][ridx])])

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph
