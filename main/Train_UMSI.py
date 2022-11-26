import numpy as np
import matplotlib.pyplot as plt
import sys, os
import tensorflow as tf
from keras.optimizers import Adam
import cv2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from PIL import Image
from copy import deepcopy
import tqdm
import math, random

sys.path.append('../src')

from data_loading import load_datasets_singleduration
from util import get_model_by_name, create_losses

from losses_keras2 import *
from sal_imp_utilities import *
from cb import InteractivePlot
from losses_keras2 import loss_wrapper

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# FILL THESE IN 
bp = "/netpool/homes/wangyo/Dataset/"

dataset_imp = "imp1k"
dataset_sal = "UMSI_SALICON"
data_imp = load_datasets_singleduration(dataset_imp, bp)
data_sal = load_datasets_singleduration(dataset_sal, bp)

# FILL THESE IN: set training parameters 
ckpt_savedir = "ckpt"

load_weights = False
weightspath = "./ckpt/weights.hdf5"

batch_size = 4
init_lr = 0.0001
lr_reduce_by = .1
reduce_at_epoch = 3
n_epochs = 15

opt = Adam(lr=init_lr) 

model_name = "UMSI"
model_inp_size = (240, 320)
model_out_size = (480, 640)

input_shape = model_inp_size + (3,)


# get model 
model_params = {
    'input_shape': input_shape,
    'n_outs': 2
}
model_func, mode = get_model_by_name(model_name)
#assert mode == "simple", "%s is a multi-duration model! Please use the multi-duration notebook to train." % model_name
model = model_func(**model_params)

if load_weights: 
    model.load_weights(weightspath)
    print("load")



# set up data generation and checkpoints
if not os.path.exists(ckpt_savedir): 
    os.makedirs(ckpt_savedir)

# Generators
gen_train = ImpAndClassifGenerator(
        img_filenames=data_imp[0],
        imp_filenames=data_imp[1],
        fix_filenames=None,
        extra_imgs=data_sal[0], # For feeding a much larger dataset, e.g. salicon, that the generator will subsample to maintain class balance
        extra_imps=data_sal[1],
        extra_fixs=None,
        extras_per_epoch=160,
        batch_size=4,
        img_size=(shape_r,shape_c),
        map_size=(shape_r_out, shape_c_out),
        shuffle=True,
        augment=False,
        n_output_maps=1,
        concat_fix_and_maps=False,
        fix_as_mat=False,
        fix_key="",
        str2label=None,
        dummy_labels=False,
        num_classes=6,
        pad_imgs=True,
        pad_maps=True,
        return_names=False,
        return_labels=True,
        read_npy=False)

gen_val = ImpAndClassifGenerator(
            img_filenames=data_imp[3], 
            imp_filenames=data_imp[4], 
            fix_filenames=None, 
            extra_imgs=data_sal[3], # For feeding a much larger dataset, e.g. salicon, that the generator will subsample to maintain class balance
            extra_imps=data_sal[4],
            extra_fixs=None,
            extras_per_epoch=40,
            batch_size=1, 
            img_size=(shape_r,shape_c), 
            map_size=(shape_r_out, shape_c_out),
            shuffle=False, 
            augment=False, 
            str2label=None,
            dummy_labels=False,
            #n_output_maps=1,
        )

# Callbacks

# where to save checkpoints
filepath = os.path.join(ckpt_savedir, dataset_imp + '_kl+cc+bin_ep{epoch:02d}_valloss{val_loss:.4f}.hdf5')

print("Checkpoints will be saved with format %s" % filepath)

cb_chk = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=1)
cb_plot = InteractivePlot()

def step_decay(epoch):
    lrate = init_lr * math.pow(lr_reduce_by, math.floor((1+epoch)/reduce_at_epoch))
    if epoch%reduce_at_epoch:
        print('Reducing lr. New lr is:', lrate)
    return lrate
cb_sched = LearningRateScheduler(step_decay)

cbs = [cb_chk, cb_sched, cb_plot]



#test the generator 
img, outs = gen_train.__getitem__(1)
print("batch size: %d. Num inputs: %d. Num outputs: %d." % (batch_size, len(img), len(outs)))
print(outs[0].shape)
print(outs[1].shape)

model.compile(optimizer=opt, loss={'dec_c_cout': kl_cc_combined, "out_classif":"binary_crossentropy"}, loss_weights={'dec_c_cout': 1, "out_classif":5})

print('Ready to train')
model.fit_generator(gen_train, epochs=n_epochs, verbose=1, callbacks=cbs, validation_data=gen_val, max_queue_size=10, workers=5)