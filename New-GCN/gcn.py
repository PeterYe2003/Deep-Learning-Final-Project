from conv_layer import *
from metrics import m_softmax_cross_entropy, m_accuracy
from config import args


class GCN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, num_nonzero_features, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nonzero_features = num_nonzero_features
        self.conv_layers = []

        self._build_layers()

    def __call__(self, inputs, training=None, mask=None, **kwargs):
        return self.call(inputs, training, mask)

    def _build_layers(self):

        self.conv_layers.append(GraphConvolutionLayer(input_dim=self.input_dim,
                                                 output_dim=args.hidden1,
                                                 num_nonzero_features=self.num_nonzero_features,
                                                 act_fn=tf.nn.relu,
                                                 dropout=args.dropout,
                                                 has_sparse_inputs=True
                                                 ))
        self.conv_layers.append(GraphConvolutionLayer(input_dim=args.hidden1,
                                                 output_dim=self.output_dim,
                                                 num_nonzero_features=self.num_nonzero_features,
                                                 act_fn=lambda x: x,
                                                 dropout=True
                                                 ))

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: Union[tf.bool, bool] = None, mask: tf.Tensor =None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Executes a forward pass through the model and computes loss and accuracy.

        :param inputs: A tuple containing input features, target labels, and adjacency matrix.

        :param training: Whether the model is in training mode. (Default is True)

        :param mask: Mask to apply on the labels. (Default is None)

        :returns: A tuple containing loss and accuracy.


        .. note::
            This function processes the input data through the model's layers in a forward pass.
            It calculates the loss using softmax cross-entropy with optional masking and L2 regularization.
            The loss value is accumulated by adding the L2 norm of each trainable variable multiplied by the weight decay.
            Accuracy is computed based on model predictions and the provided labels.
        """

        features, label, adj = inputs
        outputs = [features]
        for layer in self.conv_layers:
            hidden_outputs = layer(inputs=(outputs[-1], adj), training=training)
            outputs.append(hidden_outputs)
        out = outputs[-1]

        loss = zeros_tensor([])
        for var in self.conv_layers[0].trainable_variables:
            loss += args.weight_decay * tf.nn.l2_loss(var)
        loss += m_softmax_cross_entropy(out, label, mask)
        acc = m_accuracy(out, label, mask)

        return loss, acc

    def predict(self):
        return tf.nn.softmax(self.outputs)
    