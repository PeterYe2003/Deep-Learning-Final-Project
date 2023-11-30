from conv_layer import *
from metrics import m_softmax_cross_entropy, m_accuracy
from config import args


class GCN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, num_nonzero_features, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nonzero_features = num_nonzero_features
        self.layers = []

        self._build_layers()

    def __call__(self, inputs, training=None, mask=None, **kwargs):
        return self.call(inputs, training, mask)

    def call(self, inputs, training=None, mask=None):
        features, label, support = inputs
        outputs = [features]
        for layer in self.layers:
            hidden_outputs = layer((outputs[-1], support), training)
            outputs.append(hidden_outputs)
        out = outputs[-1]

        loss = 0
        for var in self.layers_[0].trainable_variables:
            loss += args.weight_decay * tf.nn.l2_loss(var)
        loss += m_softmax_cross_entropy(out, label, mask)
        acc = m_accuracy(out, label, mask)

        return loss, acc

    def _build_layers(self):

        self.layers.append(GraphConvolutionLayer(input_dim=self.input_dim,
                                                 output_dim=args.hidden1,
                                                 num_nonzero_features=self.num_nonzero_features,
                                                 act_fn=tf.nn.relu,
                                                 dropout=args.dropout,
                                                 sparse_inputs=True
                                                 ))
        self.layers.append(GraphConvolutionLayer(input_dim=args.hidden1,
                                                 output_dim=self.output_dim,
                                                 num_nonzero_features=self.num_nonzero_features,
                                                 act_fn=lambda x: x,
                                                 dropout=True
                                                 ))

    def _predict(self):
        return tf.nn.softmax(self.outputs)

