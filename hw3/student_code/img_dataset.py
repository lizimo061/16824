from torch.utils.data import Dataset
import torch
from external.vqa.vqa import VQA
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import h5py
from tqdm import tqdm
from pathlib import Path

class ImgDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern, existing_format=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
            """
        self.image_dir = image_dir
        self.question_json_file_path = question_json_file_path
        self.annotation_json_file_path = annotation_json_file_path
        self.image_filename_pattern = image_filename_pattern

        # self.existing_format = existing_format

        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.queIds = self.vqa.getQuesIds();
        self.img_list = self.getUniqueImg()

    def imgIdToPath(self, idx):
        str_bracket = "{}"
        start_idx = self.image_filename_pattern.find(str_bracket)
        path = self.image_filename_pattern[0:start_idx] + str(idx).zfill(12) + self.image_filename_pattern[start_idx+2:]
        path = self.image_dir + "/" + path
        return path

    def getUniqueImg(self):
        count_img = []
        for i in tqdm(range(len(self.queIds))):
            qa_id = self.queIds[i]
            qa = self.vqa.loadQA(qa_id)[0]
            image_id = qa['image_id']
            if image_id not in count_img:
                    count_img.append(image_id)
        print("Unique images size: ", len(count_img))
        return count_img


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if idx >= len(self.vqa.qa):
            print("Error: access overflow")
            return None

        img_idx = self.img_list[idx]
        data = {}

        data['images_id'] = img_idx
        tmp_img = Image.open(self.imgIdToPath(img_idx))
        tmp_img = tmp_img.convert('RGB')
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

        trans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
            ])
        data['images'] = trans(tmp_img)

        return data

