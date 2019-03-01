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

    def call_pool5(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        return x

    def call_fc7(self, inputs, training=False):
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
        return out

    def call_fc7_pool5(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x_conv5 = self.conv5(x)
        x = self.pool3(x_conv5)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out,training=training)
        out = self.dense2(out)
        return x_conv5,out

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

    # train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
    #                                                              class_names=CLASS_NAMES,
    #                                                              split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.batch(args.batch_size)

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    model.build((args.batch_size,224,224,3))
    ckpt_path = "./tb/2019-02-25_10-45-32/"
    status = checkpoint.restore(os.path.join(ckpt_path,"ckpt-60"))
    status.assert_consumed()

    # Test visualization
    # tmp = test_images[4,:,:,:]
    # plt.imshow(tmp)
    # plt.show()
    # return

    #0,1,2,3,6,7,10 20 22 25
    #1 2 3 4 10 11 15 40 45 54 image name
    query_ind = [0,1,2,3,6,7,10,20,22,25] # For testing only, need to generate them for each class
    image_num = test_images.shape[0]
    total_pool5_out = []
    total_fc7_out = []
    for batch, (images, labels, weights) in enumerate(test_dataset):
        pool5_out,fc7_out = model.call_fc7_pool5(images)
        pool5_out = pool5_out.numpy()
        pool5_out = pool5_out.reshape((pool5_out.shape[0], pool5_out.shape[1]*pool5_out.shape[2]*pool5_out.shape[3]))
        #fc7_out = model.call_fc7(test_images)
        fc7_out = fc7_out.numpy()
        for i in range(pool5_out.shape[0]):
            total_pool5_out.append(pool5_out[i,:])
            total_fc7_out.append(fc7_out[i,:])
    total_pool5_out = np.array(total_pool5_out)
    total_fc7_out = np.array(total_fc7_out)

    # pool5_out = model.call_pool5(test_images)
    # pool5_out = pool5_out.numpy()
    # pool5_out = pool5_out.reshape((image_num, pool5_out.shape[1]*pool5_out.shape[2]*pool5_out.shape[3]))
    kdt = KDTree(total_pool5_out, metric='euclidean')
    pool5_inds = kdt.query(total_pool5_out[np.array(query_ind)], k=5,return_distance=False)
    # fc7_out = model.call_fc7(test_images)
    # fc7_out = fc7_out.numpy()
    print(pool5_inds)

    kdt = KDTree(total_fc7_out, metric='euclidean')
    fc7_inds = kdt.query(total_fc7_out[np.array(query_ind)], k=5,return_distance=False)
    print(fc7_inds)
    # For visualization
    for i in range(0,len(query_ind)):
        img_list_pool5 = pool5_inds[i,:]
        img_list_fc7 = fc7_inds[i,:]
        img_name_pool5 = "./hw1/figures/caffe_pool5_" + str(i)
        img_name_fc7 = "./hw1/figures/caffe_fc7_" + str(i)
        for j in range(1,5):
            img_id = img_list_pool5[j]
            save_name = img_name_pool5 + "_" + str(j) + ".jpg"
            img = test_images[img_id,:,:,:]
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.savefig(save_name)

        for j in range(1,5):
            img_id = img_list_fc7[j]
            save_name = img_name_fc7 + "_" + str(j) + ".jpg"
            img = test_images[img_id,:,:,:]
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.savefig(save_name)








if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
