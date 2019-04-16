import torch.nn as nn
import torch.nn.functional as F

class AttentionNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.init = 0
		self.x_embedding = nn.Sequential(
			nn.Linear(input_ques_size, embed_size),
			nn.Tanh(),
			nn.Dropout(0.5))

		self.g_embedding = nn.Sequential(
			nn.Linear(input_ques_size, embed_size), # TODO size wrong 
			nn.Tanh(),
			nn.Dropout(0.5))

		self.hx_weights = nn.Lienar(embed_size,1)
		self.softmax = nn.SoftMax()

	def forward(self,image_feat, ques_feat):
		# Three steps:
		# Ques_feat size D*T

		# First step:
		feat = self.x_embedding(ques_feat)
		#feat: K*T
		h1 = self.hx_weights(feat)
		#h1: 1*T
		p1 = self.softmax(h1)
		#p1: 1*T
	 	p1 = p1.view(-1,1)
	 	quesAtt1 = torch.mm(ques_feat,p1)
	 	#quesAtt1: D*1

	 	# Second step:
	 	x_feat = self.x_embedding(image_feat)
	 	g_feat = self.g_embedding(quesAtt1)

	 	added = x_feat + g_feat

	 	h2 = self.hx_weights(added)
	 	p2 = self.softmax(h2)

	 	image_atten = p2.view(-1,1)
	 	visAtt = torch.mm(image_feat, image_atten)




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
        self.embed_size

        self.lang_embedding = nn.Sequential(
        	nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size),
        	nn.Tanh(),
        	nn.Dropout(0.5),
        	)
        # Embedding layer output B*T*D

        #Transpose to B*D*T

        self.unigram = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernal_size=1, padding=0)
        self.bigram = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernal_size=2, padding=1)
        self.trigram = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernal_size=3, padding=1)

        # Output by these layers should be B*D*T
        # Remember to narrow bigram by:
        # self.bigram = bigram.narrow(1,0,self.seq_size) seq_size is T 

        # Get max from 3 B*D*T

        self.max = nn.MaxPool1d(kernal_size=3)

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size)


    def forward(self, image, question_encoding):
        # TODO

        # If preprocessed:
        images_feat = images.contiguous().view(-1,2048,196)



