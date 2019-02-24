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

def test(dataset,model):
    test_loss = tfe.metrics.Mean()

    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)
        test_loss(loss_value)

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
    img_save_interval = 200

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,                                                                             split='test')
    ## TODO modify the following code to apply data augmentation here
    print("======== Loading done ========")
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

    checkpoint_dir = os.path.join(logdir, "ckpt")
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    ## TODO write the training and testing code for multi-label classification
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
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

            if global_step.numpy() % args.log_interval == 0:

                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}  '.format(ep,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         epoch_loss_avg.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())

                # Tensorboard Visualization
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('training_loss', epoch_loss_avg.result())
                    tf.contrib.summary.scalar('learning_rate', learning_rate())
                    for grad,var in zip(grads,model.trainable_variables):
                        tf.contrib.summary.histogram("gradients_{0}".format(var.name), grad)

            if global_step.numpy() % args.eval_interval == 0:
                with tf.contrib.summary.always_record_summaries():
                    test_AP, test_mAP = util.eval_dataset_map(model, test_dataset)
                    tf.contrib.summary.scalar('test_map', test_mAP)
                    test_loss = test(test_dataset,model)
                    tf.contrib.summary.scalar('testing_loss', test_loss)

            if global_step.numpy() % img_save_interval == 0:
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.image('training_img', images)
    # Save checkpoints
    checkpoint.save(file_prefix=checkpoint_dir)
    AP, mAP = util.eval_dataset_map(model, test_dataset)

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
