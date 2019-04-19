from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch

class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, preprocessing):

        train_h5_path = "./features/resnet/train/train_feat_resnet_{}.h5"
        test_h5_path = "./features/resnet/test/test_feat_resnet_{}.h5"

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg", existing_format=None, prepro=preprocessing, prepro_path=train_h5_path)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg", existing_format=train_dataset, prepro=preprocessing, prepro_path=test_h5_path)

        embed_size = 512
        vocab_size = train_dataset.quesVecSize
        ans_size = train_dataset.ansVecSize
        seq_len = train_dataset.seq_len

        self._model = CoattentionNet(embed_size, vocab_size, ans_size, seq_len)

        super().__init__(train_dataset, val_dataset, self._model, batch_size, num_epochs,
                         num_data_loader_workers, preprocessing)

        # self.optimizer = torch.optim.RMSprop(self._model.parameters(), lr=4e-4, momentum=0.99, weight_decay=1e-8) 
        # self.optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-8)
        self.optimizer = torch.optim.Adam(self._model.parameters())

    def _optimize(self, predicted_answers, true_answer_ids):
        # TODO
        self.optimizer.zero_grad()

        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.NLLLoss()
        loss = criterion(predicted_answers, true_answer_ids)
        loss.backward()
        self.optimizer.step()

        return loss

