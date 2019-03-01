from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import eager as tfe
from tensorflow.keras import layers
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors
import util

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='SimpleCNN')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=96,
                                   kernel_size=[11, 11],
                                   strides=4,
                                   padding="valid",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3),strides=2)
        self.conv2 = layers.Conv2D(filters=256,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(3, 3),strides=2)
        self.conv3 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv4 = layers.Conv2D(filters=384,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv5 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=(3, 3),strides=2)
        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)

        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out,training=training)
        out = self.dense2(out)
        out = self.dropout2(out,training=training)
        out = self.dense3(out)

        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

    def get_conv_weights(self):
        return self.conv1.weights

def test(dataset,model):
    test_loss = tfe.metrics.Mean()

    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)
        tess_loss(loss_value)

    return test_loss.result()

def kernel_normalize(weights):
    max_val = np.amax(weights)
    min_val = np.amin(weights)
    norm_weights = 255*(weights-min_val)/(max_val-min_val)
    return norm_weights


def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval-interval', type=int, default=20,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log-dir', type=str, default='tb',
                        help='path for logging directory')
    parser.add_argument('--data-dir', type=str, default='./VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')

    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    # logdir = os.path.join(args.log_dir,
    #                       datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # if os.path.exists(logdir):
    #     shutil.rmtree(logdir)
    # os.makedirs(logdir)
    

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    model.build((args.batch_size,224,224,3))
    ckpt_path = "./tb/2019-02-25_10-45-32/"

    for cp_ind in range(1,61,5):
        status = checkpoint.restore(os.path.join(ckpt_path,"ckpt-"+str(cp_ind)))
        weights = model.get_weights()
        status.assert_consumed()

        conv_weights = model.get_conv_weights()
        kernel_weights = conv_weights[0].numpy() # 11 11 3 96

        visualize_idx = [0,10,20]
        for i in visualize_idx:
            kernel_weight = kernel_weights[:,:,:,i]
            norm = colors.Normalize(0,1)
            norm_weight = kernel_normalize(kernel_weight)
            plt.imshow(norm_weight,cmap='gray')
            img_name = "./hw1/figures/ckpt-" + str(cp_ind) + "_conv1_f" + str(i) + ".jpg"
            # plt.show()
            plt.savefig(img_name)

            # Using PIL, not correct tho
            # conv_img = Image.fromarray(kernel_weight, 'RGB')
            # img_name = "./hw1/figures/ckpt-" + str(cp_ind) + "_conv1_f" + str(i) + ".jpg"
            # conv_img.save(img_name)
            # conv_img.show()


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
