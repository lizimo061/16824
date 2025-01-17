import argparse
import os
import torch
import shutil
import time
from datetime import datetime
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics
from logger import Logger

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from datasets.factory import get_imdb
from custom import *

import pdb
import traceback

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float32)

    denorm = transforms.Normalize((-mean/std).tolist(), (1.0/std).tolist())

    return denorm(image)


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    log_path = os.path.join("./log/",datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_path)
    logger = Logger(log_path,'http://localhost','8097',use_visdom=True)

    torch.manual_seed(6)
    np.random.seed(6)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, trainval_imdb, logger=None)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    # logger = None
    # if args.vis:
    #     logger = Logger("./log/",'http://localhost','8097')


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, trainval_imdb, logger)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, trainval_imdb, logger)
            logger.scalar_summary("validate/metric1",m1,epoch)
            logger.scalar_summary("validate/metric2",m2,epoch)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)




#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, db, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    batch_num = len(train_loader)
    img_record_inverval= batch_num/3
    img_record_per_batch = 2
    # switch to train mode
    model.train()

    end = time.time()


    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        iter_num = epoch*batch_num + i + 1

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input # bs, 3, 512, 512
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``

        output = model(input) # 20,20,29,29
        # Global max-pooling

        imoutput = F.max_pool2d(output, kernel_size=output.shape[2]) # 20,20,1,1
        imoutput = imoutput.view(-1, imoutput.shape[1]) # 20,20
        imoutput = torch.sigmoid(imoutput)
        loss = criterion(imoutput, target)


        

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        logger.scalar_summary("train/loss",losses.avg,iter_num)
        logger.scalar_summary("train/metric1",avg_m1.avg,iter_num)
        logger.scalar_summary("train/metric2",avg_m2.avg,iter_num)

        # TODO:
        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for tag, params in model.named_parameters():
        #     if params.grad is None:
        #         continue
        #     tag = tag.replace('.','/')
        #     weights = params.data
        #     gradients = params.grad.data
        #     logger.hist_summary(tag, weights, iter_num)
        #     logger.hist_summary(tag+'/grad', gradients, iter_num)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals

        if i % img_record_inverval==0 and logger!=None:
            for ind in range(img_record_per_batch):
                img = denormalize(input_var[ind,:,:,:])
                # img = input_var[ind,:,:,:]
                tmp_heat = output[ind,:,:,:]
                gt_class = target[ind,:]
                img = img.data.numpy()
                tmp_heat = tmp_heat.data.cpu().numpy()
                gt_class = gt_class.data.cpu().numpy()
                gt_class = np.nonzero(gt_class)[0].astype(int)
                tmp_heat = tmp_heat[gt_class,:,:]

                # For original image
                title = str(epoch) + "_" + str(iter_num) + "_" + str(i) + "_image_" + str(ind)
                logger.img_summary("train/img/"+ title,img,iter_num)
                img_vis = (img - np.min(img))/(np.max(img)-np.min(img))
                img_vis = np.uint8(img_vis*255)
                logger.vis_img(img_vis, title)

                #For heat map
                for cla in range(tmp_heat.shape[0]):
                    heat = tmp_heat[cla,:,:]
                    title = str(epoch) + "_" + str(iter_num) + "_" + str(i) + "_heatmap_" + db.classes[gt_class[cla]] + "_" + str(ind)

                    heat = (heat - np.min(heat))/(np.max(heat)-np.min(heat))

                    heat_trans = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((img.shape[1],img.shape[2]))
                        ])
                    heat = torch.Tensor(heat).unsqueeze(0)
                    heat_map = heat_trans(heat)
                    heat_map = np.uint8(cm.jet(np.array(heat_map))*255)
                    heat_map = np.transpose(heat_map, axes=(2,0,1))

                    logger.img_summary("train/heat_map/"+ title,heat_map,iter_num)

                    logger.vis_img(heat_map, title)


        # End of train()


def validate(val_loader, model, criterion,db,logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``

        output = model(input) 
        # Global max-pooling
        imoutput = F.max_pool2d(output, kernel_size=output.shape[2]) # 20,20,1,1
        imoutput = imoutput.view(-1, imoutput.shape[1]) # 20,20
        imoutput = F.sigmoid(imoutput)
        loss = criterion(imoutput, target)



        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))
        # if logger!=None:
        #     logger.scalar_summary("validate/metric1",avg_m1.avg,i)
        #     logger.scalar_summary("validate/metric2",avg_m2.avg,i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals

        if i<20 and logger!=None:
            img = denormalize(input_var[0,:,:,:])
            tmp_heat = output[0,:,:,:]
            gt_class = target[0,:]
            img = img.data.numpy()
            tmp_heat = tmp_heat.data.cpu().numpy()
            gt_class = gt_class.data.cpu().numpy()
            gt_class = np.nonzero(gt_class)[0].astype(int)
            tmp_heat = tmp_heat[gt_class,:,:]
                
            # For original image
            title = "validate_" + str(i) + "_image" 
            logger.img_summary("validate/img/" + title,img,i)
            img_vis = (img - np.min(img))/(np.max(img)-np.min(img))
            img_vis = np.uint8(img_vis*255)
            logger.vis_img(img_vis, title)

            for cla in range(tmp_heat.shape[0]):
                heat = tmp_heat[cla,:,:]
                title = "validate_" + str(i) + "_heatmap_" + db.classes[gt_class[cla]]
                    
                heat = (heat - np.min(heat))/(np.max(heat)-np.min(heat))

                heat_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((img.shape[1],img.shape[2]))
                    ])
                heat = torch.Tensor(heat).unsqueeze(0)
                heat_map = heat_trans(heat)
                heat_map = np.uint8(cm.jet(np.array(heat_map))*255)
                heat_map = np.transpose(heat_map, axes=(2,0,1))

                logger.img_summary("validate/heat_map/" + title,heat_map,i)
                    
                logger.vis_img(heat_map, title)




    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    # mAP
    nclasses = target.shape[1]
    AP = []
    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for cid in range(nclasses):
        gt_cls = target[:,cid].astype('float32')
        pred_cls = output[:,cid].astype('float32')

        if len(np.nonzero(gt_cls)[0])!=0:
            pred_cls -= 1e-5 * gt_cls
            ap = sklearn.metrics.average_precision_score(gt_cls,pred_cls)
            AP.append(ap)
        else:
            ap = 0
        

    mAP = sum(AP)/len(AP)
    return [mAP]


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    score = []
    nclasses = target.shape[1]
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    
    for i in range(target.shape[0]):
        output_1d = output[i,:]
        target_1d = target[i,:]
        output_1d = np.where(output_1d>0.01,1,0)
        score.append(sklearn.metrics.f1_score(target_1d,output_1d,average='weighted'))

    return [np.mean(score)]


if __name__ == '__main__':
    main()
