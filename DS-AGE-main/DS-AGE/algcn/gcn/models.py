import sys

sys.path.append('/Users/brandondu/Documents/GitHub/Deep-Learning-Final-Project/DS-AGE-main/DS-AGE/algcn/gcn')
from layers import *
from metrics import *
from configuration import *

class GCN:
    def __init__(self, placeholders, input_dim, **kwargs):
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.layers = []
        self.activations = []
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.opt_op = None

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)

        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += args.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def build(self):
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=args.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True
                                            ))

        self.layers.append(GraphConvolution(input_dim=16,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True
                                            ))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class Simple_GCN:
    def __init__(self, placeholders, input_dim, **kwargs):
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.layers = []
        self.activations = []
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.opt_op = None

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)

        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += args.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def build(self):
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=args.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=True
                                            ))

        self.layers.append(GraphConvolution(input_dim=args.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True
                                            ))

    def predict(self):
        return tf.nn.softmax(self.outputs)
