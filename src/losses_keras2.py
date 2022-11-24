import keras.backend as K
import numpy as np
from sal_imp_utilities import *
from tensorflow.keras.losses import KLDivergence

# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=1), axis=1)
    sum_x = K.sum(K.sum(y_true, axis=1), axis=1)
    sum_y = K.sum(K.sum(y_pred, axis=1), axis=1)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=1), axis=1)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=1), axis=1)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return num / den


def kl_new(y_true, y_pred):
    '''
    This function is for singleduration model. The old kl_divergence() may cause nan in training.
    '''
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())
    kl = tf.keras.losses.KLDivergence()
    return kl(y_true,y_pred)


def kl_cc_combined(y_true, y_pred):
    # For Singleduration
    '''Loss function that combines cc, nss and kl. Beacuse nss receives a different ground truth than kl and cc (maps),
        the function requires y_true to contains both maps. It has to be a tensor with dimensions [bs, 2, r, c, 1]. y_pred also
        has to be a tensor of the same dim, so the model should add a 5th dimension between bs and r and repeat the predict map
        twice along that dim.
    '''

    #k = kl_time(y_true, y_pred)
    k = kl_new(y_true, y_pred)
    print('k=',k)
    #c = cc_time(y_true, y_pred)
    c = correlation_coefficient(y_true, y_pred)
    print('c=', c)
    return 10*k-3*c


def loss_wrapper(loss, input_shape):
    shape_r_out, shape_c_out = input_shape
    print("shape r out, shape c out", shape_r_out, shape_c_out)
    def _wrapper(y_true, y_pred):
        return loss(y_true, y_pred)
    return _wrapper
