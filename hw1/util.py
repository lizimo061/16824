import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image


def set_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    return session


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_pascal(data_dir, class_names, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        class_names (list): list of class names
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 256px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    ## TODO Implement this function
    img_dir = os.path.join(data_dir, 'JPEGImages')
    imgset_dir = os.path.join(data_dir, 'ImageSets','Main')

    print "Reading from ", data_dir

    image_list = open(os.path.join(imgset_dir, split+".txt"), "r").read()
    image_name = image_list.split('\n')[:-1]
    image_num = len(image_name)

    images = np.zeros((image_num,256,256,3), dtype=np.float32)
    labels = np.zeros((image_num,20), dtype=np.int32)
    weights = np.zeros((image_num,20), dtype=np.int32)

    for i in xrange(image_num):
        # Resize image
        image = Image.open(os.path.join(img_dir, image_name[i]+".jpg"))
        print os.path.join(img_dir, image_name[i]+".jpg")
        image = image.resize((256,256))
        image_np = np.array(image)
        images[i,:,:,:] = image_np[np.newaxis,:,:,:]

        for j,each_class in enumerate(class_names):
            class_file = os.path.join(imgset_dir, each_class+"_"+split+".txt")
            class_data = np.loadtxt(class_file, dtype=str)
            image_loc = np.where(class_data == image_name[i])
            image_label = class_data[np.asscalar(image_loc[0])][1]

            if image_label == '1':
                labels[i,j] = 1
                weights[i,j] = 1
            else:
                labels[i,j] = 0
                weights[i,j] = 1
                if image_label == '0':
                    weights[i,j] = 0

    return images, labels, weights


def cal_grad(model, loss_func, inputs, targets, weights=1.0):
    """
    Return the loss value and gradients
    Args:
         model (keras.Model): model
         loss_func: loss function to use
         inputs: image inputs
         targets: labels
         weights: weights of the samples
    Returns:
         loss and gradients
    """

    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss_func(targets, logits, weights)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    gt = []
    pred = []
    valid = []
    for batch, (images, labels, weights) in enumerate(dataset):
        predictions = model(images)
        for i, logits in enumerate(preictions):
            p = tf.nn.softmax(logits)
            gt.append(p)
            pred.append(labels[i,:])
            valid.append(weights[i,:])
    gt = np.array(gt)
    pred = np.array(pred)
    valid = np.array(valid)

    AP = compute_ap(gt, pred, valid)
    mAP = AP/float(gt.shape[1])
    ## TODO implement the code here

    return AP, mAP


def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr

# For debugging
if __name__ == '__main__':
    data_dir = '/home/zimol/Downloads/16824_data/VOCdevkit/VOC2007'
    class_names = ['aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',]
    load_pascal(data_dir, class_names, split='train')
