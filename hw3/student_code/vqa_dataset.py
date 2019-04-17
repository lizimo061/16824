from torch.utils.data import Dataset
import torch
from external.vqa.vqa import VQA
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import h5py
import tqdm
class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern, existing_format=None, ques_thres=12, ans_thres=6, seq_len=50, prepro=False, prepro_path=None):
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
        self.prepro = prepro
        print("Allow preprocessing: ", self.prepro)
        # self.existing_format = existing_format

        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.queIds = self.vqa.getQuesIds();
        self.quesWords = self.getSplitQues()
        self.ansWords = self.getSplitAns()

        self.ques_thres = ques_thres
        self.ans_thres = ans_thres

        self.id_images = {}

        if existing_format is None:
            self.quesWordToIdx, self.quesVecSize = self.BuildBoW(self.quesWords,self.ques_thres)
            self.ansWordToIdx, self.ansVecSize = self.BuildBoW(self.ansWords,self.ans_thres)
            # self.quesVecSize = len(self.quesWordToIdx)
            self.seq_len = seq_len

        else:
            self.quesWordToIdx = existing_format.quesWordToIdx
            self.ansWordToIdx = existing_format.ansWordToIdx
            self.quesVecSize = existing_format.quesVecSize
            self.ansVecSize = existing_format.ansVecSize
            self.seq_len = existing_format.seq_len

        if self.prepro:
        	self.prepro_path = prepro_path
        	# features_extracted = h5py.File(prepro_path, 'r')
        	# self.features_h5 = features_extracted["features"][:]
        	# self.ids_h5 = features_extracted["ids"][:]

        print("The length of question vector: ", self.quesVecSize)
    
    def getImgSize(self):
    	return len(self.vqa.getImgIds())

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
        vocab_set['NA'] = 0
        for i in range(0,len(data)):
            for vocab in data:
                if vocab not in vocab_set:
                    idx = len(vocab_set)
                    vocab_set[vocab] = idx

        return vocab_set

    def BuildBoW(self, data, thres):
        vocab_set = {}
        for vocab in data:
            if vocab not in vocab_set:
                vocab_set[vocab] = 1
            else:
                vocab_set[vocab] += 1

        vocab_map = {}
        vocab_map['NA'] = 0
        idx = 1
        for vocab in data:
            if vocab not in vocab_map:
                if vocab_set[vocab] > thres:
                    vocab_map[vocab] = idx
                    idx += 1
                else:
                    vocab_map[vocab] = vocab_map['NA']

        # vocab_map['END'] = -1

        return vocab_map, idx

    def BoWVoting(self, sentences, table):
    	count_set = {}
    	for word in sentences:
    		if word['answer'] not in count_set:
    			count_set[word['answer']] = 1
    		else:
    			count_set[word['answer']] += 1
    	sorted_dict = sorted(count_set.items(), key=lambda kv: kv[1])
    	res_word = sorted_dict[-1][0]
    	best_ind = 0
    	if res_word in table:
    		best_ind = table[res_word]
    	return np.array(best_ind)

    def BoWVector(self, sentence, table):
    	bow_vec = np.zeros(self.quesVecSize)

    	for i in range(self.seq_len):
    		if i < len(sentence):
    			if sentence[i] in table:
    				bow_vec[table[sentence[i]]] = 1
    			else:
    				bow_vec[table['NA']] = 1
    	return bow_vec

    def BoWVectorGeneric(self, sentence, table):
    	bow_vec = np.zeros([self.seq_len,self.quesVecSize])

    	for i in range(self.seq_len):
    		if i < len(sentence):
    			if sentence[i] in table:
    				bow_vec[i,table[sentence[i]]] = 1
    			else:
    				bow_vec[i,table['NA']] = 1
    		else:
    			break

    	return bow_vec

    def saveFeatures(self, feat, id):
    	self.id_images[id] = feat

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
        data['questions'] = torch.from_numpy(self.BoWVectorGeneric(tmp_question.split(),self.quesWordToIdx))
        tmp_answers = qa['answers']

        data['gt_answer'] = torch.from_numpy(self.BoWVoting(tmp_answers,self.ansWordToIdx))
        data['images_id'] = qa['image_id']
        if self.prepro:
        	# h5_idxs = self.ids_h5
        	# query_idx = np.where(h5_idxs==qa['image_id'])
        	# data['images_id'] = qa['image_id']
        	# tmp_features = self.features_h5[query_idx[0][0]]
        	# tmp_features = tmp_features.astype(np.float32)
        	# data['images'] = torch.from_numpy(tmp_features)

        	# Above for all features in one h5 file
        	# Below for several different feature files

        	img_idx = qa['image_id']
        	str_bracket = "{}"
        	start_idx = self.prepro_path.find(str_bracket)
        	path = self.prepro_path[0:start_idx] + str(img_idx) + self.prepro_path[start_idx+2:]
        	features_extracted = h5py.File(path, 'r')
        	feature = features_extracted["features"][:]
        	data['images'] = feature

        else:
        	tmp_img = Image.open(self.imgIdToPath(qa['image_id']))
        	tmp_img = tmp_img.convert('RGB')
        	normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        
        	trans = transforms.Compose([
            	transforms.Resize((224,224)),
            	transforms.ToTensor(),
            	normalize,
        	])
        	data['images'] = trans(tmp_img)

    
        
        return data

