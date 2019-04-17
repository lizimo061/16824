import torch.nn as nn
from external.googlenet.googlenet import googlenet
import torch

class SimpleBaselineNet(nn.Module):
	"""
	Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
	"""
	def __init__(self, vocab_size, embedding_size, ans_size):
		super().__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.ans_size = ans_size

		# Image features extraction
		self.img_features = googlenet(pretrained=True)

		# Word features extraction
		self.embedding = nn.Linear(self.vocab_size, self.embedding_size)
		# self.tanh = nn.Tanh()
		self.linear = nn.Linear(self.embedding_size + 1000, self.ans_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, image, question_encoding,prepro=False):
		# TODO
		question_encoding = torch.sum(question_encoding, dim=1)
		question_encoding = torch.squeeze(question_encoding, dim=1)
		image_features = image
		if prepro == False:
			image_features = self.img_features(image)
			if self.training:
				image_features = image_features[2]

		word_features = self.embedding(question_encoding)
		# word_features = self.tanh(word_features)
		features = torch.cat((image_features,word_features),1)
		features = self.linear(features)
		answers = self.softmax(features)
		return answers
