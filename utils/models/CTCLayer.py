import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras

from utils.constants import BATCH_SIZE

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost


    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.addLoss()`.
        batchLen = tf.cast(tf.shape(y_true)[0], dtype="int64")
        inputLength = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        labelLength = tf.cast(tf.shape(y_true)[1], dtype="int64")

        inputLength = inputLength * tf.ones(shape=(batchLen, 1), dtype="int64")
        labelLength = labelLength * tf.ones(shape=(batchLen, 1), dtype="int64")
        losses = self.loss_fn(y_true, y_pred, inputLength, labelLength)
        self.add_loss(losses)

        # At test time, just return the computed predictions
        return y_pred
