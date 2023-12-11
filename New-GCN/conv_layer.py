from typing import Union, Any, List, Tuple

from gcn_helpers import *


class GraphConvolutionLayer(tf.keras.layers.Layer):
    """Convolution Layer class."""

    def __init__(
            self,
            input_dim,
            output_dim,
            num_nonzero_features,
            dropout=0.0,
            has_sparse_inputs=False,
            act_fn=tf.nn.relu,
            bias=False,
            has_features=True,
            **kwargs
    ):
        super(GraphConvolutionLayer, self).__init__(**kwargs)

        self.dropout = dropout
        self.num_nonzero_features = num_nonzero_features
        self.act_fn = act_fn
        self.has_sparse_inputs = has_sparse_inputs
        self.has_features = has_features
        self.bias = bias

        w = self.add_weight(name="weight0", shape=[input_dim, output_dim])
        w.assign(gb_tensor([input_dim, output_dim]))
        self.weights_ = [w]

        if self.bias:
            b = self.add_weight(name="bias", shape=[output_dim])
            b.assign(zeros_tensor([output_dim]))

    def __call__(self, inputs):
        return self._call(inputs)

    def _dropout(self, features: Union[tf.Tensor, tf.SparseTensor], training: Any):
        """
        Apply dropout to the input tensor.

        :param features: Input tensor.

        :param training: Flag indicating whether the model is in training mode.

        :return: Output tensor after applying dropout (if in training mode).
        :rtype: tf.Tensor
        """
        if training is False:
            return features
        if self.sparse_inputs:
            features = sparse_dropout(features, self.dropout, self.num_nonzero_features)
        else:
            features = tf.nn.dropout(features, rate=self.dropout)
        return features

    def _convolve(self, features: tf.Tensor, adj_: List[tf.Tensor]):
        """
        Perform graph convolution operation on input features.

        :param features: Input features.

        :param adj_: List of adjacency matrices.

        :return: Output after graph convolution.
        """
        adj = [
            multiply_tensors(features, self.weights_[i], sparse=self.sparse_inputs)
            if self.has_features
            else self.weights_[i]
            for i in range(len(adj_))
        ]

        outputs = [
            multiply_tensors(adj_[i], adj[i], sparse=True)
            for i in range(len(adj_))
        ]

        return tf.add_n(outputs)

    def call(self, inputs: Tuple[tf.Tensor, List[tf.Tensor]], training=None, **kwargs):
        """
        Perform the computation of 4the Graph Convolutional Layer.

        :param inputs: Input data containing features and adjacency matrices.

        :param training: Whether the model is in training mode.
        :type training: bool, optional

        :param kwargs: Additional keyword arguments.

        :return: Output after the graph convolutional layer computation.
        :rtype: tf.Tensor
        """
        features, adj = inputs
        features = self._dropout(features, training)
        out = self._convolve(features, adj)
        if self.bias:
            out += self.bias
        return self.act_fn(out)
