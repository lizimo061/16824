import h5py
import torch.nn as nn
from external.googlenet.googlenet import googlenet
from student_code.vqa_dataset import VqaDataset
from student_code.img_dataset import ImgDataset

from torch.utils.data import DataLoader
import torch
from external.resnet.resnet import resnet18
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.model = googlenet(pretrained=True)

	def forward(self,x):
		x = self.model(x)
		return x

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet,self).__init__()
		self.upsample = nn.Upsample((448,448), mode='bilinear')
		self.model = resnet18(pretrained=True)
		def save_output(module, intput, output):
			self.buffer = output
		self.model.layer4.register_forward_hook(save_output)

	def forward(self,x):
		x = self.upsample(x)
		x = self.model(x)
		return self.buffer


def preprocessFromGoogleNet():

	train_image_dir = "/home/zimol/Downloads/16824_data/train2014"
	test_image_dir = "/home/zimol/Downloads/16824_data/val2014"
	train_question_path = "/home/zimol/Downloads/16824_data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json"
	test_question_path= "/home/zimol/Downloads/16824_data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json"

	test_annotation_path = "/home/zimol/Downloads/16824_data/mscoco_val2014_annotations.json"
	train_annotation_path = "/home/zimol/Downloads/16824_data/mscoco_train2014_annotations.json"

	train_dataset = VqaDataset(image_dir=train_image_dir,question_json_file_path=train_question_path,annotation_json_file_path=train_annotation_path,image_filename_pattern="COCO_train2014_{}.jpg", existing_format=None)

	test_dataset = VqaDataset(image_dir=test_image_dir,question_json_file_path=test_question_path,annotation_json_file_path=test_annotation_path,image_filename_pattern="COCO_val2014_{}.jpg", existing_format=train_dataset)

	net = Net().cuda()
	net.eval()

	batch_size = 64

	train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
	test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)


	# Train:
	features_shape = (train_dataset.getImgSize(),1000)
	train_features_path = "./features/train_feat_googlenet.h5"
	with h5py.File(train_features_path, libver='latest') as fd:
		features = fd.create_dataset('features', shape=features_shape,dtype='float32')
		ids = fd.create_dataset('ids', shape=(train_dataset.getImgSize(),),dtype='int32')

		i=j=0

		for batch_id, batch_data in enumerate(train_dataset_loader):
			images = batch_data['images'].cuda()
			out = net(images)

			j = i+images.shape[0]
			features[i:j,:] = out.data.cpu().numpy().astype('float32')
			ids[i:j] = batch_data['images_id'].numpy().astype('int32')

			i=j

		print("Save {%d} images in total for training",j)

	# Test
	features_shape = (test_dataset.getImgSize(),1000)
	test_features_path = "./features/test_feat_googlenet.h5"
	with h5py.File(test_features_path, libver='latest') as fd:
		features = fd.create_dataset('features', shape=features_shape,dtype='float32')
		ids = fd.create_dataset('ids', shape=(test_dataset.getImgSize(),),dtype='int32')

		i=j=0

		for batch_id, batch_data in enumerate(test_dataset_loader):
			images = batch_data['images'].cuda()
			out = net(images)

			j = i+images.shape[0]
			features[i:j,:] = out.data.cpu().numpy().astype('float32')
			ids[i:j] = batch_data['images_id'].numpy().astype('int32')

			i=j
		print("Save {%d} images in total for testing",j)

if __name__ == "__main__":

# def preprocessFromResNet():
	train_image_dir = "/home/zimol/Downloads/16824_data/train2014"
	test_image_dir = "/home/zimol/Downloads/16824_data/val2014"
	train_question_path = "/home/zimol/Downloads/16824_data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json"
	test_question_path= "/home/zimol/Downloads/16824_data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json"

	test_annotation_path = "/home/zimol/Downloads/16824_data/mscoco_val2014_annotations.json"
	train_annotation_path = "/home/zimol/Downloads/16824_data/mscoco_train2014_annotations.json"

	train_dataset = VqaDataset(image_dir=train_image_dir,question_json_file_path=train_question_path,annotation_json_file_path=train_annotation_path,image_filename_pattern="COCO_train2014_{}.jpg", existing_format=None)

	test_dataset = VqaDataset(image_dir=test_image_dir,question_json_file_path=test_question_path,annotation_json_file_path=test_annotation_path,image_filename_pattern="COCO_val2014_{}.jpg", existing_format=train_dataset)

	net = ResNet().cuda()
	net.eval()

	batch_size = 1
	train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
	test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

	print("Number of training images: ", len(train_dataset))
	print("Number of testing images: ", len(test_dataset))

	features_shape = (512,14,14)

	for batch_id, batch_data in enumerate(train_dataset_loader):
		images = batch_data['images'].cuda()
		out = net(images)
		idx = batch_data['images_id'].numpy().astype('int')
		train_features_path = "./features/resnet/train/train_feat_resnet_"+str(idx[0])+".h5"
		feat_file = Path(train_features_path)
		if feat_file.is_file():
			continue
		with h5py.File(train_features_path, "w") as fd:
			features = fd.create_dataset('features', shape=features_shape,dtype='float32')
			feature = out.data.cpu()
			feature = torch.squeeze(feature, dim=0)
			features[:,:,:] = feature.numpy().astype('float32')

	for batch_id, batch_data in enumerate(test_dataset_loader):
		images = batch_data['images'].cuda()
		out = net(images)
		idx = batch_data['images_id'].numpy().astype('int')
		test_features_path = "./features/resnet/test/test_feat_resnet_"+str(idx[0])+".h5"
		feat_file = Path(train_features_path)
		if feat_file.is_file():
			continue
		with h5py.File(test_features_path, "w") as fd:
			features = fd.create_dataset('features', shape=features_shape,dtype='float32')
			feature = out.data.cpu()
			feature = torch.squeeze(feature, dim=0)
			features[:,:,:] = feature.numpy().astype('float32')


	
	# Write features in one file

	# Train:

	# features_shape = (len(train_dataset),512,14,14)
	# train_features_path = "./features/train_feat_resnet.h5"
	# with h5py.File(train_features_path, libver='latest') as fd:
	# 	features = fd.create_dataset('features', shape=features_shape,dtype='float32')
	# 	ids = fd.create_dataset('ids', shape=(len(train_dataset),),dtype='int32')

	# 	i=j=0

	# 	for batch_id, batch_data in enumerate(train_dataset_loader):
	# 		images = batch_data['images'].cuda()
	# 		out = net(images)

	# 		j = i+images.shape[0]
	# 		features[i:j,:,:,:] = out.data.cpu().numpy().astype('float32')
	# 		ids[i:j] = batch_data['images_id'].numpy().astype('int32')

	# 		i=j
		

	# features_shape = (len(test_dataset),512,14,14)
	# test_features_path = "./features/test_feat_resnet.h5"
	# with h5py.File(test_features_path, libver='latest') as fd:
	# 	features = fd.create_dataset('features', shape=features_shape,dtype='float32')
	# 	ids = fd.create_dataset('ids', shape=(len(test_dataset),),dtype='int32')

	# 	i=j=0

	# 	for batch_id, batch_data in enumerate(test_dataset_loader):
	# 		images = batch_data['images'].cuda()
	# 		out = net(images)

	# 		j = i+images.shape[0]
	# 		features[i:j,:,:,:] = out.data.cpu().numpy().astype('float32')
	# 		ids[i:j] = batch_data['images_id'].numpy().astype('int32')

	# 		i=j


