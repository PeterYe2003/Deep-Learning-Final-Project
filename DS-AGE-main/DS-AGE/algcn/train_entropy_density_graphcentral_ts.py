from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy as sc
import os
import sklearn
import sys

from gcn.utils import *
from gcn.models import GCN, Simple_GCN
from gcn.configuration import *
from utils import load_randomalpdata
from utils import sample_mask
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


if args.model == 'gcn' and args.method == 'baseline' and args.dataset == 'citeseer':
    exit(0)

if args.model == 'gcn' and args.method == 'f_similarity' and args.dataset == 'citeseer':
    exit(0)

if args.model == 'gcn' and args.method == 's_similarity' and args.dataset == 'citeseer':
    exit(0)

if args.model == 'simple_gcn' and args.method == 'baseline' and args.dataset == 'citeseer':
    exit(0)

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


dataset_str = args.dataset
basef = 0
if dataset_str == 'citeseer':
    if args.method == 'all':
        basef = 0.8
    elif args.method in {'fs_similarity, fe_similarity, se_similarity'}:
        basef = 0.85
    else:
        basef = 0.9
elif dataset_str == 'cora':
    if args.method == 'all':
        basef = 0.9
    elif args.method in {'fs_similarity, fe_similarity, se_similarity'}:
        basef = 0.95
    else:
        basef = 0.99
elif dataset_str == 'pubmed':
    basef = 0.995

MAC = []
MIC = []
for index_val in ['0', '1', '2', '3', '4']:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph = load_randomalpdata(
        args.dataset, index_val, args.inicount)
    np.set_printoptions(threshold=sys.maxsize)

    message_passing = adj * adj

    raw_features = features.todense()
    features = preprocess_features(features)

    support = [preprocess_adj(adj)]
    num_supports = 1

    if args.model == 'gcn':
        model_func = GCN
    else:
        model_func = Simple_GCN

    NCL = args.num_classes
    NL = NCL * 20
    tf.compat.v1.disable_eager_execution()

    placeholders = {
        'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.compat.v1.placeholder(tf.int32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.compat.v1.Session()

    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)


    def evaluatepred(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.predict()], feed_dict=feed_dict_val)
        predlabels = np.argmax(outs_val[1], axis=1)
        return outs_val[0], predlabels, (time.time() - t_test)

    def perc(input_list, k):
        count = sum(1 for i in input_list if i < input_list[k])
        return count / len(input_list)


    def percd(input_list, k):
        count = sum(1 for i in input_list if i > input_list[k])
        return count / len(input_list)


    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())

    cost_val = []

    normcen = np.loadtxt("res/" + args.dataset + "/graphcentrality/normcen")
    cenperc = np.asarray([perc(normcen, i) for i in range(len(normcen))])

    # Train model
    for epoch in range(args.epochs):

        t = time.time()

        gamma = np.random.beta(1, 1.005 - basef ** epoch)
        if args.method == 'baseline':
            alpha = beta = delta = epsilon = phi = (1 - gamma) / 2
        elif args.method in {'fs_similarity, fe_similarity, se_similarity'}:
            alpha = beta = delta = epsilon = phi = (1 - gamma) / 4
        elif args.method == 'all':
            alpha = beta = delta = epsilon = phi = (1 - gamma) / 5
        else:
            alpha = beta = delta = epsilon = phi = (1 - gamma) / 3

        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)

        feed_dict.update({placeholders['dropout']: args.dropout})
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict(), model.outputs], feed_dict=feed_dict)

        if len(idx_train) < NL:
            curr_features = raw_features[idx_train, :]
            curr_features = sklearn.preprocessing.normalize(np.asarray(curr_features))
            raw_features = sklearn.preprocessing.normalize(np.asarray(raw_features))
            similarity = []
            for i in curr_features:
                similarity.append(np.dot(raw_features, np.squeeze([i])))
            similarity = np.squeeze(np.array(similarity))
            max_similarity = np.max(similarity, axis=0)
            simprec = np.asarray([percd(max_similarity, i) for i in range(len(max_similarity))])

            curr_embeddings = outs[4][idx_train, :]
            curr_embeddings = sklearn.preprocessing.normalize(curr_embeddings)
            raw_embeddings = sklearn.preprocessing.normalize(outs[4])
            em_similarity = []
            for i in curr_embeddings:
                em_similarity.append(np.dot(raw_embeddings, np.squeeze([i])))
            em_similarity = np.squeeze(np.array(em_similarity))
            max_em_similarity = np.max(em_similarity, axis=0)
            em_simprec = np.asarray([percd(max_em_similarity, i) for i in range(len(max_em_similarity))])

            connectivity = message_passing[idx_train, :]
            max_connectivity = np.squeeze(np.array(np.max(connectivity, axis=0).todense()))
            connprec = np.asarray([percd(max_connectivity, i) for i in range(len(max_connectivity))])

            entropy = sc.stats.entropy(outs[3].T)
            train_mask = sample_mask(idx_train, labels.shape[0])
            entrperc = np.asarray([perc(entropy, i) for i in range(len(entropy))])

            kmeans = KMeans(n_clusters=NCL, random_state=0, n_init=10).fit(outs[3])
            ed = euclidean_distances(outs[3], kmeans.cluster_centers_)
            ed_score = np.min(ed, axis=1)
            edprec = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])

            if args.method == 'baseline':
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc
                print("entropy weight: ", alpha, " density weight: ", beta, "centrality weight: ", gamma)
            elif args.method == 'f_similarity':
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * simprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " feature similarity weight: ", delta)
            elif args.method == 's_similarity':
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * connprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " structural similarity weight: ", delta)
            elif args.method == 'e_similarity':
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * em_simprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " embedding similarity weight: ", delta)
            elif args.method == 'fs_similarity': # FDS SDS
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * simprec + epsilon * connprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " feature similarity weight: ", delta, " structural similarity weight: ", epsilon)
            elif args.method == 'fe_similarity':  # FDS EDS
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * simprec + epsilon * em_simprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " feature similarity weight: ", delta, " embedding similarity weight: ", epsilon)
            elif args.method == 'se_similarity': # EDS SDS
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * connprec + epsilon * em_simprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " structural similarity weight: ", delta, " embedding similarity weight: ", epsilon)
            else: # ALL
                finalweight = alpha * entrperc + beta * edprec + gamma * cenperc + delta * connprec + epsilon * simprec + phi * em_simprec
                print("entropy weight: ", alpha, " density weight: ", beta, " centrality weight: ", gamma,
                      " structural similarity weight: ", delta, " feature similarity weight: ", epsilon, " embedding similarity weight: ", phi)
            finalweight[train_mask + val_mask + test_mask] = -100
            select = np.argmax(finalweight)
            idx_train.append(select)
            train_mask = sample_mask(idx_train, labels.shape[0])
            y_train = np.zeros(labels.shape)
            y_train[train_mask, :] = labels[train_mask, :]
        else:
            print('finish select!')

        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping + 1):-1]) and len(
                idx_train) >= NL:
            print("Early stopping...")
            break

    print("Optimization Finished!")

    test_cost, y_pred, test_duration = evaluatepred(features, support, y_test, test_mask, placeholders)
    y_true = np.argmax(y_test, axis=1)[test_mask]
    macrof1 = f1_score(y_true, y_pred[test_mask], average='macro')
    microf1 = f1_score(y_true, y_pred[test_mask], average='micro')
    print(macrof1, microf1)
    directory = "res/" + args.dataset + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.model == 'gcn':
        f = open(directory + "val_" + args.method + "_ini_" + str(args.inicount) + "_macrof1.txt", "a")
        f.write("{:.5f}\n".format(macrof1))
        f.close()
        f1 = open(directory + "val_" + args.method + "_ini_" + str(args.inicount) + "_microf1.txt", "a")
        f1.write("{:.5f}\n".format(microf1))
        f1.close()
    else:
        f = open(directory + "simple_gcn_" + "val_" + args.method + "_ini_" + str(args.inicount) + "_macrof1.txt", "a")
        f.write("{:.5f}\n".format(macrof1))
        f.close()
        f1 = open(directory + "simple_gcn_" + "val_" + args.method + "_ini_" + str(args.inicount) + "_microf1.txt", "a")
        f1.write("{:.5f}\n".format(microf1))
        f1.close()


    MAC.append(macrof1)
    MIC.append(microf1)

print(MAC)
print(MIC)

print("mean of macrof1 over 5 runs: ", np.mean(MAC))
print("mean of microf1 over 5 runs: ", np.mean(MIC))
print("variance of macrof1 over 5 runs: ", np.var(MAC))
print("variance of microf1 over 5 runs: ", np.var(MIC))
