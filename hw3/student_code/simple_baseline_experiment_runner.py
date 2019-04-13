from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, preprocessing):

        train_h5_path = "./features/train_feat_googlenet.h5"
        test_h5_path = "./features/test_feat_googlenet.h5"

        embedding_size = 1024

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg", existing_format=None, prepro=preprocessing, prepro_path=train_h5_path)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg", existing_format=train_dataset, prepro=preprocessing, prepro_path=test_h5_path)

        model = SimpleBaselineNet(vocab_size=train_dataset.quesVecSize, embedding_size=embedding_size, ans_size=train_dataset.ansVecSize)

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers, preprocessing)

        self.optimizer = torch.optim.SGD([
                                {'params': model.embedding.parameters(), 'lr': 0.8},
                                {'params': model.softmax.parameters()},{'params':model.linear.parameters()}], lr=1e-2, momentum=0.9) 
        # TODO_private

    def _optimize(self, predicted_answers, true_answer_ids):
        # TODO
        self.optimizer.zero_grad()

        #Getting loss here
        criterion = torch.nn.NLLLoss()

        # print(predicted_answers[0:])
        # print(true_answer_ids)

        loss = criterion(predicted_answers, true_answer_ids)

        loss.backward()
        self.optimizer.step()  

        return loss      
