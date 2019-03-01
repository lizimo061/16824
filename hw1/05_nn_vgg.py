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
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from PIL import Image
from matplotlib import pyplot as plt
import util

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='SimpleCNN')
        self.num_classes = num_classes
        self.conv1_1 = layers.Conv2D(input_shape=(224,224,3),
                                     filters=64,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv1_2 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))

        self.pool1 = layers.MaxPool2D(pool_size=(2, 2),strides=2)

        self.conv2_1 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv2_2 = layers.Conv2D(filters=128,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))

        self.pool2 = layers.MaxPool2D(pool_size=(2, 2),strides=2)

        self.conv3_1 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv3_2 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv3_3 = layers.Conv2D(filters=256,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))

        self.pool3 = layers.MaxPool2D(pool_size=(2, 2),strides=2)

        self.conv4_1 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv4_2 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv4_3 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))

        self.pool4 = layers.MaxPool2D(pool_size=(2, 2),strides=2)

        self.conv5_1 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv5_2 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.conv5_3 = layers.Conv2D(filters=512,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))

        self.pool5 = layers.MaxPool2D(pool_size=(2, 2),strides=2)
        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)

        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

    def call_pool5(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        return x

    def call_fc7(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)

        return out

def test(dataset,model):
    test_loss = tfe.metrics.Mean()

    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)
        test_loss(loss_value)

    return test_loss.result()

def img_mean_substract(img):
    h = img.shape[0]
    w = img.shape[1]
    mean_mask = np.empty(img.shape)
    R = np.full((h,w),123.68)
    G = np.full((h,w),116.78)
    B = np.full((h,w),103.94)
    mean_mask[:,:,0] = R
    mean_mask[:,:,1] = G
    mean_mask[:,:,2] = B
    result = img - mean_mask
    return result

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
    img_save_interval = 200

    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    model.build((args.batch_size,224,224,3))
    ckpt_path = "./tb/2019-02-25_23-00-46/"
    status = checkpoint.restore(os.path.join(ckpt_path,"ckpt-1"))
    status.assert_consumed()

    # Test visualization
    # tmp = test_images[1,:,:,:]
    # plt.imshow(tmp)
    # plt.show()
    # return


    query_ind = [0,1,2,3,6,7,10,20,22,25] # For testing only, need to generate them for each class
    image_num = test_images.shape[0]
    
    pool5_out = model.call_pool5(test_images)
    pool5_out = pool5_out.numpy()
    pool5_out = pool5_out.reshape((image_num, pool5_out.shape[1]*pool5_out.shape[2]*pool5_out.shape[3]))
    kdt = KDTree(pool5_out, leaf_size=30, metric='euclidean')
    pool5_inds = kdt.query(pool5_out[np.array(query_ind)], k=5,return_distance=False)
    fc7_out = model.call_fc7(test_images)
    fc7_out = fc7_out.numpy()
    kdt = KDTree(fc7_out, leaf_size=30, metric='euclidean')
    fc7_inds = kdt.query(fc7_out[np.array(query_ind)], k=5,return_distance=False)

    # For visualization
    for i in range(0,len(query_ind)):
        img_list_pool5 = pool5_inds[i,:]
        img_name_pool5 = "./hw1/figures/vgg16_pool5_" + str(i)
        img_name_fc7 = "./hw1/figures/vgg16_fc7_" + str(i)
        for j in range(1,5):
            img_id = pool5_inds[0][j]
            save_name = img_name_pool5 + "_" + str(j) + ".jpg"
            img = test_images[img_id,:,:,:]
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.savefig(save_name)

        for j in range(1,5):
            img_id = fc7_inds[0][j]
            save_name = img_name_fc7 + "_" + str(j) + ".jpg"
            img = test_images[img_id,:,:,:]
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.savefig(save_name)


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
