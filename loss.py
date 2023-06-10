import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def weighted_BCE_loss(y_true, y_pred):
    '''
    computes weighted bce loss
    '''
    total = y_true.shape[1] + y_true.shape[2] + y_true.shape[3]
    pos = np.sum(y_true)
    weights = K.ones_like(y_true)
    weights = tf.where(y_true == 1, (total / pos) * weights, weights)
    out = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    out = K.expand_dims(out, axis=-1) * weights 
    return K.mean(out)