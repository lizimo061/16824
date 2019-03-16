import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index
    #If you did Task 0, you should know how to set these values from the imdb

    classes = list(imdb.classes)
    class_to_idx = {}
    for ind,item in enumerate(classes,1):
        class_to_idx[item] = ind

    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset
    image_num = imdb.num_images
    dataset_list = []
    gt_roidb = imdb.gt_roidb()
    for i in range(image_num):
        img_path = imdb.image_path_at(i)
        gt_classes = gt_roidb[i]['gt_classes']
        dataset_list.append((img_path, gt_classes))

    return dataset_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64,192,kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192,384,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,num_classes,kernel_size=1,stride=1),
        )

    def forward(self, x):
        #TODO: Define forward pass
        x = self.features(x)
        x = self.classifier(x)

        return x

    # For heat map
    # def conv_output(self,x):
    #     conv_out = self.features(x)
    #     x = self.classifier(conv_out)

    #     return conv_out,x


class LocalizerAlexNetHighres(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetHighres, self).__init__()
        #TODO: Ignore for now until instructed









    def forward(self, x):
        #TODO: Ignore for now until instructed










        return x



def localizer_alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or
    # not
    if pretrained:
        model_dict = model.state_dict()
        pre_state_dict = model_zoo.load_url(model_urls['alexnet'])

        pre_dict = {k:v for k,v in pre_state_dict.items() if k in model_dict and 'features' in k}
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        print("====== Pretrained model loaded ======")
    else:
        print("====== Initialize from scratch ======")
        for m in model.features.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    for m in model.classifier.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    return model





def localizer_alexnet_robust(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed








    return model




class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write this function, look at the imagenet code for inspiration
        img_path = self.imgs[index][0]
        img_cls = self.imgs[index][1]
        img = self.loader(img_path)
        img = self.transform(img)

        target = np.zeros(len(self.classes))
        target[img_cls-1] = 1

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
