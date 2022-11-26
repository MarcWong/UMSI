import numpy as np
import keras
from keras.layers import Layer, Input, Multiply, Dropout, TimeDistributed, LSTM, Activation, Lambda, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, BatchNormalization, Concatenate, Add, DepthwiseConv2D
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from dcn_resnet_new import dcn_resnet
from gaussian_prior_new import LearningPrior
from sal_imp_utilities import *
from xception_custom import Xception_wrapper
from keras.regularizers import l2



def decoder_block(x, dil_rate=(2,2), print_shapes=True, dec_filt=1024):
    # Dilated convolutions
    x = Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//4, 3, padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = Conv2D(dec_filt//4, 3, padding='same', activation='relu', dilation_rate=(2, 2))(x)
    x = UpSampling2D((4,4), interpolation='bilinear')(x)
    if print_shapes: print('Shape after last ups:',x.shape)

    # Final conv to get to a heatmap
    x = Conv2D(1, kernel_size=1, padding='same', activation='relu')(x)
    if print_shapes: print('Shape after 1x1 conv:',x.shape)

    return x

def decoder_block_simple(x, dil_rate=(2,2), print_shapes=True, dec_filt=1024):

    x = Conv2D(dec_filt, 3, padding='same', activation='relu')(x)
    x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu')(x)
    x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//4, 3, padding='same', activation='relu')(x)
    x = UpSampling2D((4,4), interpolation='bilinear')(x)
    if print_shapes: print('Shape after last ups:',x.shape)

    # Final conv to get to a heatmap
    x = Conv2D(1, kernel_size=1, padding='same', activation='relu')(x)
    if print_shapes: print('Shape after 1x1 conv:',x.shape)

    return x


def decoder_block_dp(x, dil_rate=(2,2), print_shapes=True, dec_filt=1024, dp=0.3):
    # Dilated convolutions
    x = Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=dil_rate)(x)
    x = Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=dil_rate)(x)
    x = Dropout(dp)(x)
    x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=dil_rate)(x)
    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=dil_rate)(x)
    x = Dropout(dp)(x)
    x = UpSampling2D((2,2), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//4, 3, padding='same', activation='relu', dilation_rate=dil_rate)(x)
    x = Dropout(dp)(x)
    x = UpSampling2D((4,4), interpolation='bilinear')(x)

    x = Conv2D(dec_filt//4, 3, padding='same', activation='relu', dilation_rate=dil_rate)(x)
    x = Dropout(dp)(x)
    x = UpSampling2D((4,4), interpolation='bilinear')(x)
    if print_shapes: print('Shape after last ups:',x.shape)

    # Final conv to get to a heatmap
    x = Conv2D(1, kernel_size=1, padding='same', activation='relu')(x)
    if print_shapes: print('Shape after 1x1 conv:',x.shape)

    return x



######### ENCODER DECODER MODELS #############

def xception_decoder(input_shape = (shape_r, shape_c, 3),
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                    dil_rate = (2,2)):

    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception:',xception.output.shape)

    ## DECODER ##
    outs_dec = decoder_block(xception.output, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=512)

    outs_final = [outs_dec]*n_outs

    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m


def resnet_decoder(input_shape = (shape_r, shape_c, 3),
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                    dil_rate = (2,2)):
    inp = Input(shape=input_shape)

    ### ENCODER ###
    dcn = dcn_resnet(input_tensor=inp)
    if print_shapes: print('resnet output shape:',dcn.output.shape)

    ## DECODER ##
    outs_dec = decoder_block(dcn.output, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=512)

    outs_final = [outs_dec]*n_outs

    # Building model
    m = Model(inp, outs_final)
    if verbose:
        m.summary()
    return m


def fcn_vgg16(input_shape=(shape_r, shape_c, 3),
                 verbose=True,
                 print_shapes=True,
                 n_outs=1,
                 ups=8,
                 dil_rate=(2,2),
                 freeze_enc=False,
                 freeze_cl=True,
                 internal_filts=256,
                 num_classes=4,
                 dp=0.3,
                 weight_decay=0.,
                 batch_shape=None):

    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    print("pool5 shape", x.shape)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)

    # classification layer from fc7
    classif_layer_fc7 = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='linear',
                padding='valid', strides=(1, 1))(x)
    print("classif_layer_fc7 shape", classif_layer_fc7.shape)

    # Upsampling fc7 classif layer to sum with pool4 classif layer
    classif_layer_fc7_ups = UpSampling2D(size=(2,2), interpolation="bilinear")(classif_layer_fc7)
    print("classif_layer_fc7_ups shape", classif_layer_fc7_ups.shape)

    # Lambda layer to match shape of pool4
    def concat_one(fc7):
        shape_fc7 = K.shape(fc7)
        shape_zeros = (shape_fc7[0], 1, shape_fc7[2],  shape_fc7[3] )
        return K.concatenate([K.zeros(shape=shape_zeros), classif_layer_fc7_ups], axis=1)
    classif_layer_fc7_ups = Lambda(concat_one)(classif_layer_fc7_ups)
    print("classif_layer_fc7_ups shape after lambda:", classif_layer_fc7_ups.shape)

    # Classification layer from pool4
    classif_layer_pool4 = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='linear',
                padding='valid', strides=(1, 1))(pool4)

    x = Add()([classif_layer_pool4, classif_layer_fc7_ups])

    outs_up = UpSampling2D(size=(32, 32), interpolation="bilinear")(x)

    outs_final = [outs_up]*n_outs

    model = Model(img_input, outs_final)

    weights_path = '../../predimportance_shared/models/ckpt/fcn_vgg16/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)

    if verbose:
        model.summary()

    return model


############# UMSI MODELS ###############

def UMSI(input_shape = (shape_r, shape_c, 3),
                     conv_filters=256,
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                     freeze_enc=False,
                     return_sequences=False):
    inp = Input(shape = input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
	        layer.trainable = False
    
    # ASPP
    # TODO: Fill the missing parameters in
    c0 = Conv2D(name = "aspp_csep0")(xception.output)
    c6 = DepthwiseConv2D(name="aspp_csepd6_depthwise")(xception.output)
    c12 = DepthwiseConv2D(name="aspp_csepd12_depthwise")(xception.output)
    c18 = DepthwiseConv2D(name="aspp_csepd18_depthwise")(xception.output)

    
    c6 = BatchNormalization(name="aspp_csepd6_depthwise_BN")(c6)
    c12 = BatchNormalization(name="aspp_csepd12_depthwise_BN")(c12)
    c18 = BatchNormalization(name="aspp_csepd18_depthwise_BN")(c18)
    c6 = Activation(name = "activation_2")(c6)
    c12 = Activation(name = "activation_4")(c12)
    c18 = Activation(name = "activation_6")(c18)
    c6 = Conv2D(name = "aspp_csepd6_pointwise")(c6)
    c12 = Conv2D(name = "aspp_csepd12_pointwise")(c12)
    c18 = Conv2D(name = "aspp_csepd18_pointwise")(c18)

    c0 = BatchNormalization(name='aspp0_BN')(c0)
    c6 = BatchNormalization(name='aspp_csepd6_pointwise_BN')(c6)
    c12 = BatchNormalization(name='aspp_csepd12_pointwise_BN')(c12)
    c18 = BatchNormalization(name='aspp_csepd18_pointwise_BN')(c18)

    c0 = Activation(name = "aspp0_activation")(c0)
    c6 = Activation(name = "activation_3")(c6)
    c12 = Activation(name = "activation_5")(c12)
    c18 = Activation(name = "activation_7")(c18)

    concat1 = Concatenate(name="concatenate_1")([c0,c6,c12,c18])


    ### classification module ###
    # TODO: Fill the missing layers in
    classif = some_functions(xception.output)


    out_classif = Dense(6, activation="softmax", name="out_classif")(classif)


    x = Dense(256, name="dense_fusion")(classif)


    def lambda_layer_function(x):
        x = tf.reshape(x,(tf.shape(x)[0],1,1,256))
        con = [x for i in range(30)]
        con = tf.concat(con,axis=1)
        con = tf.concat([con for i in range(40)],axis=2)
        return con
    
    
    x = Lambda(lambda_layer_function, name = "lambda_1")(x)

    concat2 = Concatenate(name="concatenate_2")([concat1, x])
    ### DECODER ###
    x = Conv2D(256,(1,1),padding="same",use_bias=False,name = "concat_projection")(concat2)
    x = BatchNormalization(name="concat_projection_BN")(x)
    x = Activation("relu", name="activation_8")(x)
    x = Dropout(.3, name="dropout_3")(x)
    x = Conv2D(256,(3,3),padding="same",use_bias=False,name = "dec_c1")(x)
    x = Conv2D(256,(3,3),padding="same",use_bias=False,name = "dec_c2")(x)
    x = Dropout(.3, name="dec_dp1")(x)
    x = UpSampling2D(size=(2,2), interpolation='bilinear', name="dec_ups1")(x)
    x = Conv2D(128,(3,3),padding="same",use_bias=False,name = "dec_c3")(x)
    x = Conv2D(128,(3,3),padding="same",use_bias=False,name = "dec_c4")(x)
    x = Dropout(.3, name="dec_dp2")(x)
    x = UpSampling2D(size=(2,2), interpolation='bilinear', name="dec_ups2")(x)
    x = Conv2D(64,(3,3),padding="same",use_bias=False,name = "dec_c5")(x)
    x = Dropout(.3, name="dec_dp3")(x)
    x = UpSampling2D(size=(4,4), interpolation='bilinear', name="dec_ups3")(x)
    out_heatmap = Conv2D(1,(1,1),padding="same",use_bias=False,name = "dec_c_cout")(x)
    
    # Building model
    outs_final = [out_heatmap, out_classif]
    print(out_heatmap.shape)
    m = Model(inp, outs_final)
    if verbose:
        m.summary()

    return m