import torch.nn as nn
from external.googlenet.googlenet import googlenet

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Image features extraction
        self.img_features = googlenet(pretrained=True)

     	# Word features extraction
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, dropout=0.2)

        self.softmax = nn.Softmax()

    def forward(self, image, question_encoding):
        # TODO
        image_features = self.img_features(image)
		embed = self.embedding(question_encoding)
		output = self.lstm(nn.utils.run.pack_padded_sequence(embed, batch_first=True))
		word_features = nn.utils.run.pack_padded_sequence(output, batch_first=True)

		features = torch.cat((image_features,word_features),1)
		answers = self.softmax(features)
		return answers
