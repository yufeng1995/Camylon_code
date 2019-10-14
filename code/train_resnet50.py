# Training pipeline for resnet50
# Author: Jin Huang
# Date: 09/20/2018 ver_1.0

import numpy as np
import keras
import sys
import os
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import model_from_json
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# sys.path.append('/home/huangjin/resources/my_camelyon/keras-multiprocess-image-data-generator')
sys.path.append('/home/huangjin/job/keras-multiprocess-image-data-generator')
# import tools.image as T
import tools.image_old as T_OLD
import multiprocessing
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.utils.multi_gpu_utils import multi_gpu_model
import argparse
from keras import metrics
from keras.optimizers import *
from keras.callbacks import TensorBoard
from utilities import LossHistory
from utilities import TensorBoardImage
import utilities
from utilities import MyCbk
from keras.applications.resnet50 import ResNet50
import resnet_factory
# import tools.threshold_choose_preprocesing as JH
import tools.mutil_processing_image_generator_balance as MB
import feature_map as FM

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


# try:
#     from importlib import reload
#     reload(T)
# except:
#     reload(T)

try:
    pool.terminate()
except:
    pass

###########################################################
            #Define the parameters#
###########################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
img_width, img_height = 224, 224
batch_size = 128
epochs = 300
classes = 2
activation = 'binary'
nb_gpu = 2
n_process = 1
pool = multiprocessing.Pool(processes=n_process)
dropout_keep_prob = 0.8
weights = "imagenet"
network = "50"
validation_format = "fix"
class_weight = False
# phase = "debug"
# log_dir = "/30TB/jin_data/tf_log/0115_res50_debug"
phase = "train"
log_dir = "/mnt/disk_share/yufeng/train_code_Camylon/old_Camylon/save/log/test/"
float_16 = False

print(log_dir)
if not log_dir:
    print("Path does not exist. Creating TensorBoard saving directory ...")
    os.mkdir(log_dir)

if float_16:
    from keras import backend as K
    print(K.floatx())
    K.set_floatx('float16')
    print(K.floatx())

#######################################################
# Provide the directory for the json files#
#######################################################
"""
For training set:
    Always use random crop and flow from the jsons.

For validation set:
    If you choose to use random crop for validation as it is in training,
    you should set the the validation_format to "random" and
    provide the path for validation json files.

    If you choose to use a fixed cropped set (image size 299*299 for inception v3 and v4),
    you should set the the validation_format to "fix" and
    provide the directory for validation images.

"""

# Path for the saving and debugging images
sample_save_path_training = "/1TB/jin_data/samples/0116"
sample_save_path_valid = "/1TB/jin_data/samples/0116"

# Original nb_sample and paths for training
# training_json_path_for_training = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/small_patch/debug/train"
training_json_path_for_training = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/middel_patch"
# validation_json_path_for_training = "/home/yufeng/resnent50/data/valid"
# validation_set_dir_for_training = "/mnt/disk_share/data/breast_cancer/lymph_node/postoperative/small_patch/keras_filter_mutil_folder_from_filter_json"
validation_set_dir_for_training = "/mnt/disk_share/data/breast_cancer/lymph_node/intraoperative_frozen/debug/224_debug/"
nb_training_samples_train_init =  3084*6 # 24632 # 268880  # 5773
# nb_random_validation_samples_train_init = 13388
nb_fix_validation_samples_train_init = 400# 64685 # 40314 24231

# Original nb_sample and paths for debugging
# training_json_path_for_debugging = "/30TB/jin_data/cam16_processed/0114_debug/json"
# validation_json_path_for_debugging = None
# validation_set_dir_for_debugging = "/30TB/jin_data/cam16_processed/valid_224/0117_filtered"

# nb_training_samples_debug_init = 14112
# nb_random_validation_samples_debug_init = None
# nb_fix_validation_samples_debug_init = 351820
# nb_fix_validation_samples_debug_init = 24497


####################################################################################
        # Adapt nb of samples for training/debug for single or multi-GPU #
####################################################################################
def calculate_nb_sample(batch_size, initial_nb_sample, nb_gpu):
    multiple = nb_gpu * batch_size
    quotient = initial_nb_sample // multiple
    nb_excess_patch = initial_nb_sample - quotient * multiple
    final_nb_sample = initial_nb_sample - nb_excess_patch

    return final_nb_sample

if phase == "debug":
    training_json_path = training_json_path_for_debugging

    if nb_gpu != 1:
        nb_training_sample = calculate_nb_sample(batch_size=batch_size,
                                                 initial_nb_sample=nb_training_samples_debug_init,
                                                 nb_gpu=nb_gpu)
    else:
        nb_training_sample = nb_training_samples_debug_init
    print("Number of training samples for debugging: %d" % nb_training_sample)

    if validation_format == "fix":
        validation_set_dir = validation_set_dir_for_debugging

        if nb_gpu != 1:
            nb_validation_sample = calculate_nb_sample(batch_size=batch_size,
                                                     initial_nb_sample=nb_fix_validation_samples_debug_init,
                                                     nb_gpu=nb_gpu)
        else:
            nb_validation_sample = nb_fix_validation_samples_debug_init
        print("Number of fix validation samples for debugging: %d" % nb_validation_sample)

    elif validation_format == "random":
        validation_json_path = validation_json_path_for_debugging

        if nb_gpu != 1:
            nb_validation_sample = calculate_nb_sample(batch_size=batch_size,
                                                       initial_nb_sample=nb_random_validation_samples_debug_init,
                                                       nb_gpu=nb_gpu)
        else:
            nb_validation_sample = nb_random_validation_samples_debug_init
        print("Number of random validation samples for debugging: %d" % nb_validation_sample)

    else:
        print("You have to choose a validation format between fix and random!")
        sys.exit(0)

elif phase == "train":
    training_json_path = training_json_path_for_training

    if nb_gpu != 1:
        nb_training_sample = calculate_nb_sample(batch_size=batch_size,
                                                 initial_nb_sample=nb_training_samples_train_init,
                                                 nb_gpu=nb_gpu)
    else:
        nb_training_sample = nb_training_samples_train_init
    print("Number of training samples for training: %d" % nb_training_sample)

    if validation_format == "fix":
        validation_set_dir = validation_set_dir_for_training

        if nb_gpu != 1:
            nb_validation_sample = calculate_nb_sample(batch_size=batch_size,
                                                       initial_nb_sample=nb_fix_validation_samples_train_init,
                                                       nb_gpu=nb_gpu)
        else:
            nb_validation_sample = nb_fix_validation_samples_train_init
        print("Number of fix validation samples for training: %d" % nb_validation_sample)

    elif validation_format == "random":
        validation_json_path = validation_json_path_for_training

        if nb_gpu != 1:
            nb_validation_sample = calculate_nb_sample(batch_size=batch_size,
                                                       initial_nb_sample=nb_random_validation_samples_train_init,
                                                       nb_gpu=nb_gpu)
        else:
            nb_validation_sample = nb_random_validation_samples_train_init
        print("Number of random validation samples for debugging: %d" % nb_validation_sample)

    else:
        print("You have to choose a validation format between fix and random!")
        sys.exit(0)

else:
    print("You have to choose between debug or train!")
    sys.exit(0)


###########################################################
                #Multi-GPU settings#
###########################################################
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=nb_gpu,
                help="# of GPUs to use for training")
args = vars(ap.parse_args())
# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]


###########################################################
                        #Select model#
##########################################################
"""
Model options: "50", "se_50", "101", "152"

""" 
if network == "50":
    base_model = ResNet50(include_top=False, weights="imagenet")
elif network == "se_50":
    base_model = resnet_factory.se_ResNet50(include_top=False, weights=None)
elif network == "101":
    base_model = resnet_factory.ResNet101(include_top=False, weights=None)
elif network == "152":
    base_model = resnet_factory.ResNet152(include_top=False, weights=None)
else:
    print("Please select a valid network for training!")
    sys.exit(0)

base_model_out = base_model.output
base_model_out = keras.layers.GlobalAveragePooling2D()(base_model_out)

###########################################################
                        #Build model#
###########################################################
# Choose logistic layer according to the loss function.
if activation == 'binary': # Default binary
    predictions = Dense(1, activation='sigmoid')(base_model_out)
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        # model = Model(inputs=base_model.input, outputs=predictions)
        # model.summary()
        # import pdb
        # pdb.set_trace()
        # model.get_weights("/home/zhangyufeng/resnent_50/pretrain_model/postoperative_resnet50_0724_model_and_weights_epoch_8.hdf5")
        # model = keras.models.load_model("/home/zhangyufeng/resnent_50/pretrain_model/postoperative_resnet50_0724_model_and_weights_epoch_8.hdf5")
        parallel_model = model
        parallel_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    else:
        print(("[INFO] training with {} GPUs...".format(G)))
        with tf.device('/cpu:0'):
            model = Model(inputs=base_model.input, outputs=predictions)
            # model = keras.models.load_model("/mnt/disk_share/yufeng/train_code_Camylon/old_Camylon/pretrain_model/postoperative_resnet50_0822_model_and_weights_epoch_11.hdf5") # 5200 + hard_normal
            # model = keras.models.load_model("/home/zhangyufeng/resnent_50/pretrain_model/1_postoperative_resnet50_0625_model_and_weights_epoch_8.hdf5") # GPU4
            # model = keras.models.load_model("/home/zhangyufeng/model/pre_train/1_postoperative_resnet50_0625_model_and_weights_epoch_8.hdf5")  # GPU1
            # model = keras.models.load_model("/home/zhangyufeng/resnent_50/model/7_24/postoperative_resnet50_0625_model_and_weights_epoch_8.hdf5") #1W
            # model = keras.models.load_model("/home/zhangyufeng/resnent_50/model/7_29/postoperative_resnet50_0729_model_and_weights_epoch_8.hdf5") # 5200
            model = keras.models.load_model("/mnt/disk_share/yufeng/train_code_Camylon/old_Camylon/log_model/model/postoperative_resnet50_0729_model_and_weights_epoch_8.hdf5") # 5200
            # import pdb
            # pdb.set_trace()
            # model.load_weights("/home/zhangyufeng/resnent_50/model/7_24/postoperative_resnet50_0625_model_and_weights_epoch_8.hdf5")
            # print("11")
            for i in range(len(model.layers)):
                # import pdb
                # pdb.set_trace()
                if i > len(model.layers) - 35:
                # if i > len(model.layers) - 1:
                    layer = model.layers[i].trainable = True
                else:
                    layer = model.layers[i].trainable = False
        parallel_model = multi_gpu_model(model, gpus=G)
        parallel_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

elif activation == 'softmax': # Default softmax
    print("Using normal softmax cross entropy as loss ...")
    predictions = Dense(2, activation='softmax')(base_model_out)
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        model = Model(inputs=base_model.input, outputs=predictions)
        # model.summary()
        # print model.summary()
        parallel_model = model
        parallel_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=[metrics.categorical_accuracy])
    else:
        print(("[INFO] training with {} GPUs...".format(G)))
        with tf.device('/cpu:0'):
            model = Model(inputs=base_model.input, outputs=predictions)

        parallel_model = multi_gpu_model(model, gpus=G)
        parallel_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=[metrics.categorical_accuracy])

else:
    print("You have to choose the activation between sigmoid and softmax!")
    sys.exit(0)
print("********************training**************************")



###########################################################
                    #Data generator#
###########################################################
# Data Generator for training (always use json files)

# labels = ['tumor', 'normal','hard_normal',"hard_tumor","lymphatic_sinusoid"]
labels = ['tumor', 'normal',"hard_tumor"]
nb_per_class = [58, 48, 30]
# labels = ['tumor', 'normal']
# nb_per_class = [50, 48, 30]
foreground_rate_per_class = [0.7, 0.1, 0.001]
# foreground_rate_per_class = [0.5, 0.1, 0.001]
# labels = ["lymphatic_sinusoid","hard_tumor"]
# nb_per_class = [64,64]
# foreground_rate_per_class = [0.1,0.1]
# labels = ['hard_normal',"hard_tumor"]
# nb_per_class = [64, 64]
# foreground_rate_per_class = [0.01, 0.01]

train_datagen = MB.ImageDataGenerator(rescale=1. / 255,
                                        # shear_range=0.2,
                                        # zoom_range=0.1,
                                        # rotation_range = 50,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        pool = pool)

training_generator = train_datagen.flow_from_json(
                                    training_json_path ,
                                    labels = labels ,
                                    nb_per_class = nb_per_class,
                                    foreground_rate_per_class = foreground_rate_per_class,
                                    target_size=(img_height, img_width),
                                    batch_size=batch_size,
                                    classes=classes,
                                    shuffle=True,
                                    save_to_dir=sample_save_path_training,
                                    class_mode='binary',
                                    nb_gpu=nb_gpu,
                                    is_training=True)

# Data Generator for validation
if validation_format == "random":
    valid_datagen = T_OLD.ImageDataGenerator(rescale=1./255,
                                         pool=pool)

    validation_generator = valid_datagen.flow_from_json(
                                        validation_json_path,
                                        target_size=(img_height, img_width),
                                        batch_size=batch_size,
                                        class_mode='binary',
                                        shuffle=False,
                                        nb_gpu=nb_gpu)

elif validation_format == "fix":
    # labels = ["hard_normal","hard_tumor"]
    labels = ["normal","tumor"]
    nb_per_class = [40,24] # [40454:24231].
    # nb_per_class = [29,99] # [82:285]
    # nb_per_class = [58,70] # [82:68]
    foreground_rate_per_class = [0.001,0.001]
    valid_datagen = MB.ImageDataGenerator(rescale=1. / 255,
                                        # shear_range=0.2,
                                        # zoom_range=0.1,
                                        # rotation_range = 50,
                                        horizontal_flip = False,
                                        vertical_flip = False,
                                        pool = pool)

    validation_generator = train_datagen.flow_from_json(
                                        validation_set_dir,
                                        labels = labels ,
                                        nb_per_class = nb_per_class,
                                        foreground_rate_per_class = foreground_rate_per_class,
                                        target_size=(img_height, img_width),
                                        batch_size=batch_size,
                                        classes=classes,
                                        shuffle=False,
                                        save_to_dir=sample_save_path_training,
                                        class_mode='binary',
                                        nb_gpu=nb_gpu,
                                        is_training = False)

    # valid_datagen = T_OLD.ImageDataGenerator(rescale=1. / 255,
    #                                          pool=pool)

    # validation_generator = valid_datagen.flow_from_directory(
    #                                     directory=validation_set_dir,
    #                                     target_size=(img_height, img_width),
    #                                     batch_size=batch_size,
    #                                     class_mode='binary',  # only data, no labels
    #                                     shuffle=True,
    #                                     save_to_dir=sample_save_path_valid,
    #                                     seed=None,
    #                                     nb_gpu=nb_gpu)  # keep data in same order as labels

else:
    print("You have to choose a validation format between fix or random!")
    sys.exit(0)

###########################################################
                    #Callback functions#
###########################################################
loss_history = LossHistory()
lrate = LearningRateScheduler(utilities.step_decay)
cbk = MyCbk(model)

tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=128,
                            write_graph=True, write_grads=False, write_images=False,
                            embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

# fp_callback=FM.MyTensorBoard(log_dir=log_dir,input_images=training_generator.__next__(), batch_size=batch_size,
                             # update_features_freq=64,write_features=True,write_graph=True,update_freq='batch')


callbacks_list = [loss_history, lrate, cbk, tb_callback]

###########################################################
                        #Training#
###########################################################
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if class_weight:
    # Set your class weight here!
    class_weight = {0:1., 1:65.40}

    print("=>Start training with class weights ...")
    print("Choice of loss:" + loss_fun)
    parallel_model.fit_generator(generator=training_generator,
                        steps_per_epoch=nb_training_sample//(batch_size*nb_gpu),
                        validation_data=validation_generator,
                        epochs=epochs,
                        use_multiprocessing=False,
                        workers=1,
                        validation_steps=nb_validation_sample//(batch_size),
                        verbose=1,
                        class_weight=class_weight,
                        callbacks=callbacks_list)
else:
    print("=>Start training without class weight...")
    print("Choice of activation:" + activation)
    parallel_model.fit_generator(generator=training_generator,
                    steps_per_epoch=nb_training_sample//(batch_size),
                    # validation_data=validation_generator,
                    epochs=epochs,
                    use_multiprocessing=False,
                    workers=1,
                    # validation_steps=nb_validation_sample//(batch_size),
                    verbose=1,
                    callbacks=callbacks_list)

print("=> Whole training process finished.")