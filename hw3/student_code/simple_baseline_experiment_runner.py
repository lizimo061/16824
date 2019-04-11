from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
import torch.optim as optim

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg")
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg")

        model = SimpleBaselineNet()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        self.optimizer = optim.SGD([
                                {'params': model.img_features.parameters(), 'lr': 1e-2},
                                {'params': model.embedding.parameters(), 'lr': 1e-2},
                                {'params': model.softmax.parameters()}], lr=1e-3, momentum=0.9) 
        # TODO_private

    def _optimize(self, predicted_answers, true_answer_ids):
        # TODO
        self.optimizer.zero_grad()

        #Getting loss here

        loss.backward()
        self.optimizer.step()        
