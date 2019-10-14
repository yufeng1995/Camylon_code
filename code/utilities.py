# Some utilities
# Author: Jin Huang
# Initial date: 07/09/2018
# Edited: 02/20/2019 Add saving wrong prediction.

import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import LearningRateScheduler
import math
from PIL import Image
import io
import sys

###########################################################
                #Self-defined Functions #
###########################################################
def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

def step_decay(epoch):
    # if epoch >= 8:
    #     if int(epoch%2) == 0:
    #         initial_lrate = 0.0001
    #         drop = 0.5
    #         epochs_drop = 1.0
    #         lrate = initial_lrate * math.pow(drop,
    #                 math.floor((1+epoch-8)/epochs_drop))
    #     else:
    #         initial_lrate = 0.0001
    #         drop = 0.5
    #         epochs_drop = 1.0
    #         lrate = initial_lrate * math.pow(drop,
    #                 math.floor((1+epoch-8.5)/epochs_drop))
    # else:   
    #     lrate = 0.0001
    initial_lrate = 0.004
    drop = 0.5
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop,
            math.floor((1+epoch)/epochs_drop))
    if lrate < 0.0000001:
        lrate = 0.0000001
    print(lrate)
    return lrate

lrate = LearningRateScheduler(step_decay)

###########################################################
                #Self-defined Classes #
###########################################################
class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img = data.astronaut()
        # Do something to the image
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return


class MyCbk(keras.callbacks.Callback):
    def __init__(self, model, cut_off_threshold=0.7):
        self.model_to_save = model
        self.thres = cut_off_threshold

    def on_batch_end(self, epoch, batch,
                     logs={}, debug=False):
        # Save the weights for debugging.
        if debug == True:
            print("Save the weights on end of each batch.")
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            modelWeights = []
            for layer in model.layers:
                layerWeights = []
                for weight in layer.get_weights():
                    layerWeights.append(weight)
                modelWeights.append(layerWeights)
            self.weights.append(modelWeights)

            weight_dir = "/1TB/jin_data/weights"
            weights_name = "1114_debug_ms_"

            weight_save_path = weight_dir + "/" + weights_name + str(epoch)

            np.save(weight_save_path, modelWeights)

        # # Save the wrong predictions in training.
        # print("\n***********************")
        # print("Printing batch...")
        # print(logs.items())
        # sys.exit(0)

    def on_epoch_end(self, epoch, debug=False, logs=None):
        if debug == True:
            print("Saving model for debug.")
            modelWeights = []
            for layer in self.model_to_save.layers:
                layerWeights = []
                for weight in layer.get_weights():
                    layerWeights.append(weight)
                modelWeights.append(layerWeights)
            # self.weights.append(modelWeights)

            weight_dir = "/1TB/jin_data/weights"
            weights_name = "1120_ms_inv4_aux2_epoch_"

            weight_save_path = weight_dir + "/" + weights_name + str(epoch)

            try:
                np.save(weight_save_path, modelWeights)
                print("Weights saved ! (for debug)")
            except FileNotFoundError:
                print("File not found.")
                pass

        # Saving model and weights.
        if debug == True:
            model_dir = None
            model_name = None

        else:
            model_dir = "/mnt/disk_share/yufeng/train_code_Camylon/old_Camylon/save/model/test"
            print(model_dir)
            model_name = "postoperative_resnet50_0923"
            # model_name = "0222_debug_on_gpu3"
            if not model_dir:
                print("Path does not exist. Creating model saving directory ...")
                os.mkdir(model_dir)
            else:
                print("Model directory already exists.")

            print("Saving model and weights ...")
            model_full_name = model_dir + '/' + model_name + "_model_and_weights_epoch_%d.hdf5"
            self.model_to_save.save(model_full_name % epoch)
            print("Model and weights saved for epoch %d." % epoch)

            print("Saving weights only...")
            weights_full_name = model_dir + '/' + model_name + "_weights_only_epoch_%d.h5"
            self.model_to_save.save_weights(weights_full_name % epoch)
            print("Model weights only saved for epoch %d." % epoch)

class MyCbk_new(keras.callbacks.Callback):
    def __init__(self, model, cut_off_threshold=0.7):
        self.model_to_save = model
        self.thres = cut_off_threshold

    def on_batch_begin(self, epoch, batch, logs={}):
        print("\n***********************")
        print("Printing log at the start of the batch...")
        print(logs.items())

    def on_batch_end(self, epoch, batch, logs={}):
        # Save the wrong predictions in training.
        print("\n***********************")
        print("Printing log at the end of the batch...")
        print(logs.get('names', 0))
        sys.exit(0)



class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


###########################################################
                    #Self-defined Metrics#
###########################################################
def multi_metrics(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score, precision, recall


# Define a loss function for the unbalanced data
# Modified from Keras source code (losses).
def balanced_categorical_crossentropy(target, output,
                                      from_logits=False,
                                      axis=-1,
                                      positive_ratio=None):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.
    # Returns
        Output tensor.
    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))

    # Deal with the ratio of the classes
    class_weight = tf.constant([positive_ratio, 1.0 - positive_ratio])

    # Note: tf.nn.softmax_cross_entropy_with_logits expects logits
    # Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        # Add class balance to the loss
        #output = tf.mul(output, class_weight)
        return - tf.reduce_sum(target * tf.log(output), axis)

    else:
        # Manually define a balanced softmax cross entropy loss function
        weighted_logits = tf.mul(output, class_weight)
        return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=weighted_logits)



