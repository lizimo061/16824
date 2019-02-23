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

import util

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='SimpleCNN')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(filters=32,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(1024, activation='relu')
        self.dropout = layers.Dropout(rate=0.4)
        self.dense2 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)


def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch-size', type=int, default=10,
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

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')

    ## TODO modify the following code to apply data augmentation here
    ori_h = train_images.shape[1]
    ori_w = train_images.shape[2]
    crop_h = 224
    crop_w = 224
    central_fraction = 0.7

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))

    train_dataset_aug_flip = train_dataset.map(lambda img,l,w: (tf.image.random_flip_left_right(img),l,w))
    train_dataset_aug_crop = train_dataset_aug_flip.map(lambda img,l,w: (tf.random_crop(img,[crop_h,crop_w,3]),l,w))

    train_dataset.concatenate(train_dataset_aug_flip)

    test_dataset_aug = test_dataset.map(lambda img,l,w: (tf.image.central_crop(img, central_fraction),l,w))
    test_dataset_aug = test_dataset_aug.map(lambda img,l,w: (tf.image.resize_images(img,(ori_h,ori_w)),l,w))

    test_dataset.concatenate(test_dataset_aug)

    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)
    test_dataset = test_dataset.batch(args.batch_size)

    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    train_log = {'iter': [], 'loss': [], 'accuracy': []}
    test_log = {'iter': [], 'loss': [], 'accuracy': []}

    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()

        for batch, (images, labels, weights) in enumerate(train_dataset):

            loss_value, grads = util.cal_grad(model,
                                              loss_func=tf.losses.sigmoid_cross_entropy,
                                              inputs=images,
                                              weights=weights,
                                              targets=labels)
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables),
                                      global_step)
            epoch_loss_avg(loss_value)
            #debug = model(images)
            #print('labels shape {}'.format(labels.shape))
            #print('prediction shape {}'.format(debug.shape))
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('training_loss_batch', loss_value)
                test_AP, test_mAP = util.eval_dataset_map(model, test_dataset)
                tf.contrib.summary.scalar('test_map', test_mAP)

            if global_step.numpy() % args.log_interval == 0:
                # For visualization

                #tf.contrib.summary.scalar('training_loss', epoch_loss_avg.result())

                #train_AP, train_mAP = util.eval_dataset_map(model, train_dataset)
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}'.format(ep,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         epoch_loss_avg.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())
                #train_log['accuracy'].append(train_mAP)

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('training_loss_epoch', epoch_loss_avg.result())
                    #tf.contrib.summary.scalar('training_map', train_mAP)



    AP, mAP = util.eval_dataset_map(model, test_dataset)
    # For visualization
    # with tf.contrib.summary.always_record_summaries():
    #     tf.contrib.summary.scalar('testing_map', mAP)


    rand_AP = util.compute_ap(
        test_labels, np.random.random(test_labels.shape),
        test_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = util.compute_ap(test_labels, test_labels, test_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    print('Obtained {} mAP'.format(mAP))
    print('Per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, util.get_el(AP, cid)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
