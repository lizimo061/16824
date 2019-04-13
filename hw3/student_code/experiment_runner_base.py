from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
import os 
from datetime import datetime

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, preprocessing=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()
        log_path = os.path.join("./log/",datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=log_path)
        self.prepro = preprocessing
        if self._cuda:
            self._model = self._model.cuda()



    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        # TODO. Should return your validation accuracy
        correct_count = 0
        total_count = 0

        for batch_id, batch_data in enumerate(self._val_dataset_loader):
        	question_vec = batch_data['questions'].float().cuda()
        	images = batch_data['images'].cuda()
        	predicted_answer = self._model(images,question_vec, self.prepro)
        	ground_truth_answer = batch_data['gt_answer'].cuda()
        	predicted_answer_id = torch.argmax(predicted_answer, dim=1)
        	correct_item = torch.sum(torch.eq(predicted_answer_id, ground_truth_answer))
        	correct_count += correct_item.item()
        	total_count += predicted_answer.shape[0]

        	# print(correct_count)
        # print("total_count :", total_count)
        if total_count == 0:
        	return -1
        else:
        	return correct_count/float(total_count)


    def train(self):
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                question_vec = batch_data['questions'].float().cuda(async=True)
                images = batch_data['images'].cuda(async=True)

                predicted_answer = self._model(images,question_vec,self.prepro) # TODO
                ground_truth_answer = batch_data['gt_answer'] # TODO
                answer_ids = ground_truth_answer.cuda()
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, answer_ids)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    # TODO: you probably want to plot something here
                    self.writer.add_scalar("train/loss", loss, current_step)

                if current_step % self._test_freq == 0 and current_step is not 0:
                    self._model.eval()
                    with torch.no_grad():
                    	val_accuracy = self.validate()
                    	# val_accuracy = 0
                    	print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    	# TODO: you probably want to plot something here
                    	self.writer.add_scalar("test/acc", val_accuracy, current_step)
