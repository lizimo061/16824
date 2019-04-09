from torch.utils.data import Dataset
import torch
from external.vqa.vqa import VQA
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern):
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

        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.queIds = self.vqa.getQuesIds();

        self.quesWords = self.getSplitQues()
        self.ansWords = self.getSplitAns()

        self.quesWordToIdx = self.BoWPool(self.quesWords)
        self.ansWordToIdx = self.BoWPool(self.ansWords)

    def imgIdToPath(self, idx):
        str_bracket = "{}"
        start_idx = self.image_filename_pattern.find(str_bracket)
        path = self.image_filename_pattern[0:start_idx] + str(idx).zfill(12) + self.image_filename_pattern[start_idx+2:]
        path = self.image_dir + "/" + path
        return path

    def getSplitQues(self):
        ques_words = []
        for i in range(0,len(self.queIds)):
            question = self.vqa.qqa[self.queIds[i]]['question']
            question = question[0:-1]
            question = question.lower()
            ques_words += question.split()
        return ques_words

    def getSplitAns(self):
        ans_words = []
        for i in range(0,len(self.queIds)):
            anss = self.vqa.qa[self.queIds[i]]['answers']
            for ans in anss:
                ans_str = ans["answer"]
                ans_words.append(ans_str)
        return ans_words

    def BoWPool(self, data):
        vocab_set = {}

        for i in range(0,len(data)):
            for vocab in data:
                if vocab not in vocab_set:
                    idx = len(vocab_set)
                    vocab_set[vocab] = idx

        return vocab_set

    def BoWVector(self, sentence, table):
        bow_vec = np.zeros(len(table))
        for word in sentence:
            bow_vec[table[word]] = 1
        return bow_vec

    def __len__(self):
        return len(self.queIds)

    def __getitem__(self, idx):
        
        if idx >= len(self.vqa.qa):
            print("Error: access overflow")
            return None

        idx_qa = self.queIds[idx]
        qa = self.vqa.loadQA(idx_qa)[0]
        data = {}
        tmp_question = self.vqa.qqa[idx_qa]['question']
        tmp_question = tmp_question.lower()[:-1]
        data['questions'] = torch.from_numpy(self.BoWVector(tmp_question.split(),self.quesWordToIdx))
        tmp_answers = qa['answers']
        data['answers'] = np.zeros((len(tmp_answers),len(self.ansWordToIdx)))
        for i in range(0,len(tmp_answers)):
            data['answers'][i,:] = self.BoWVector([tmp_answers[i]['answer']],self.ansWordToIdx)
        data['answers'] = torch.from_numpy(data['answers'])
        tmp_img = Image.open(self.imgIdToPath(qa['image_id']))
    
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        trans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])

        data['images'] = trans(tmp_img)

        return data

