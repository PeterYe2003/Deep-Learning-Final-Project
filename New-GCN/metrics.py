import tensorflow as tf


def m_softmax_cross_entropy(preds, labels, mask):
    """
    Calculate masked softmax cross-entropy loss.

    :param preds: Predicted logits.
    :param labels: True labels.
    :param mask: Mask to apply to the loss.

    :return: Mean masked softmax cross-entropy loss
    """

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def m_accuracy(preds, labels, mask):
    """
    Calculate accuracy with masking.

    :param preds: Predicted logits
    :param labels: True labels
    :param mask: Mask to apply to accuracy computation

    :return: Mean masked accuracy
    """

    bool_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    acc = tf.cast(bool_preds, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    acc *= mask
    mean_acc = tf.reduce_mean(acc)
    return mean_acc
