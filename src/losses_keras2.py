import keras.backend as K
from sal_imp_utilities import *

# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    '''
    This function is for singleduration model.
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

    # TODO: Fill the CC loss here
    N = shape_r_out * shape_c_out
    sum_xy =
    sum_x =
    sum_y =

    num =
    den =

    return num / den


def kl_new(y_true, y_pred):
    '''
    This function is for singleduration model.
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

    k = kl_new(y_true, y_pred)
    c = correlation_coefficient(y_true, y_pred)
    print('c=', c)
    return 10*k-3*c


def loss_wrapper(loss, input_shape):
    shape_r_out, shape_c_out = input_shape
    print("shape r out, shape c out", shape_r_out, shape_c_out)
    def _wrapper(y_true, y_pred):
        return loss(y_true, y_pred)
    return _wrapper
