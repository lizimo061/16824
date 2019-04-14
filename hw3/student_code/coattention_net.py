import torch.nn as nn
import torch.nn.functional as F

class AttentionNet(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self,TODO):
		pass

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()

        self.hidden_size
        self.vocab_size
        self.seq_length
    def forward(self, image, question_encoding):
        # TODO

        # If preprocessed:


