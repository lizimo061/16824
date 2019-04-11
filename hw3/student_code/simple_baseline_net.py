import torch.nn as nn
from external.googlenet.googlenet import googlenet

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Image features extraction
        self.img_features = googlenet(pretrained=True)

     	# Word features extraction
        self.embedding = nn.Linear(self.vocab_size, self.embedding_size)

        self.softmax = nn.Softmax()

    def forward(self, image, question_encoding):
        # TODO
        image_features = self.img_features(image)
		word_features = self.embedding(question_encoding)

		features = torch.cat((image_features,word_features),1)
		answers = self.softmax(features)
		return answers
