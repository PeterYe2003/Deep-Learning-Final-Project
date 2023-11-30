from conv_layer_helpers import *
from inits import gb_tensor, zeros_tensor

class GraphConvolutionLayer(tf.keras.layers.Layer):
    """Convolution Layer class."""

    def __init__(self, input_dim, output_dim, num_nonzero_features, dropout=0.0, has_sparse_inputs=False,
                 act_fn=tf.nn.relu,
                 bias=False, has_features=True, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)

        self.dropout = dropout
        self.num_nonzero_features = num_nonzero_features
        self.act_fn = act_fn
        self.has_sparse_inputs = has_sparse_inputs
        self.has_features = has_features
        self.bias = bias

        w = self.add_weight(name='weight0', shape=[input_dim, output_dim])
        w.assign(gb_tensor([input_dim, output_dim]))
        self.weights_ = [w]
        if self.bias:
            b = self.add_weight(name='bias', shape=[output_dim])
            b.assign(zeros_tensor([output_dim]))

    def _dropout(self, x, training):
        if training is False:
            return x
        if self.sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_nonzero_features)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)
        return x

    def _convolve(self, x):

        supports = [
            multiply_tensors(x, self.weights_[i], sparse=self.sparse_inputs)
            if self.has_features
            else self.weights_[i]
            for i in range(len(self.support))
        ]

        outputs = [
            multiply_tensors(self.support[i], supports[i], sparse=True)
            for i in range(len(self.support))
        ]

        return tf.add_n(outputs)

    def call(self, inputs, *args, **kwargs):
        x, supports = inputs
        out = self._convolve(x)
        if self.bias:
            out += self.bias
        return self.act_fn(out)

    def __call__(self, inputs):
        return self._call(inputs)