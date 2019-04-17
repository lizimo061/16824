import torch.nn as nn
import torch.nn.functional as F
import torch

class MaskSoftMax(nn.Module):
	def __init__(self):
		super(MaskSoftMax, self).__init__()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x, mask):
		mask = mask.float()
		x_masked = x*mask + (1-mask)*(-10**10)
		res = self.softmax(x_masked)

		return res

class AttentionNet(nn.Module):

	def __init__(self, embed_size, seq_len, hidden_size):
		super().__init__()
		self.seq_length = seq_len
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.x_embedding = nn.Sequential(
			nn.Linear(self.embed_size, self.hidden_size),
			nn.Tanh(),
			nn.Dropout(0.5))

		self.g_embedding = nn.Sequential(
			nn.Linear(self.embed_size, self.hidden_size), # TODO size wrong 
			nn.Tanh(),
			nn.Dropout(0.5))

		self.hx_weights = nn.Linear(self.hidden_size,1)
		self.softmax = nn.Softmax()
		self.masksoftmax = MaskSoftMax()

	def forward(self,image_feat, ques_feat, mask):
		# Three steps:
		# Ques_feat size D*T
		# Image_feat size D*N e.g. D*196
		# First step:
		feat = self.x_embedding(ques_feat)
		#feat: K*T
		h1 = self.hx_weights(feat)
		#h1: 1*T
		h1 = torch.squeeze(h1,dim=2) #B*Seq
		p1 = self.masksoftmax(h1, mask) # B*Seq
		#p1: 1*T
		ques_atten = torch.unsqueeze(p1, dim=1)
		quesAtt1 = torch.matmul(ques_atten,ques_feat)
		# quesAtt1 = torch.squeeze(quesAtt1,dim=1)
	 	#quesAtt1: D*1

	 	# Second step:
		x_feat = self.x_embedding(image_feat)
		g_feat = self.g_embedding(quesAtt1)

		added = x_feat + g_feat

		h2 = self.hx_weights(added)
		p2 = self.softmax(h2)

		image_atten = p2.view(-1,1)
		visAtt = torch.mm(image_feat, image_atten)
		return quesAtt1



class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, embed_size, vocab_size, ans_size, seq_len):
        super().__init__()



        # self.hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_len
        self.embed_size = embed_size
        self.ans_size = ans_size

        self.mask = None

        self.lang_embedding = nn.Sequential(
        	nn.Linear(self.vocab_size, self.embed_size),
        	nn.Tanh(),
        	nn.Dropout(0.5),
        	)
        # Embedding layer output B*T*D

        #Transpose to B*D*T
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

        self.unigram = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernel_size=1, padding=0)
        self.bigram = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernel_size=2, padding=1)
        self.trigram = nn.Conv1d(in_channels=self.embed_size, out_channels=self.embed_size, kernel_size=3, padding=1)

        # Output by these layers should be B*D*T
        # Remember to narrow bigram by:
        # self.bigram = bigram.narrow(1,0,self.seq_size) seq_size is T 

        # Get max from 3 B*D*T

        # self.max = nn.MaxPool1d(kernal_size=3)
        # Transpose to B*T*D
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.embed_size, batch_first=True)

        self.atten_net = AttentionNet(self.embed_size, self.seq_length, self.embed_size)

    def forward(self, image, question_encoding, prepro):
        # TODO

        # question_encoding: B*Seq*vocab_size
        word_level = self.lang_embedding(question_encoding) # B*Seq*Emb_size
        word_level_for_phrase = word_level.permute(0,2,1) # B*Emb_size*Seq

        unigram = self.unigram(word_level_for_phrase)
        bigram = self.bigram(word_level_for_phrase)
        trigram = self.trigram(word_level_for_phrase) # B*Emb_size*Seq

        bigram = bigram.narrow(2,0,self.seq_length)
        
        phrase_level = torch.max(torch.max(unigram,bigram),trigram) # B*Emb_size*Seq
        phrase_level = self.tanh(phrase_level)
        phrase_level = self.dropout(phrase_level) # B*Seq*Emb_size

        phrase_for_lstm = phrase_level.permute(2,0,1)
        phrase_level = phrase_level.permute(0,2,1)

        lstm_output,(_,_) = self.lstm(phrase_for_lstm)

        ques_level = lstm_output.permute(1,2,0) # B*Emb_size*Seq
        ques_level = self.tanh(ques_level)
        ques_level = self.dropout(ques_level)

        ques_level = ques_level.permute(0,2,1) # B*Seq*Emb_size

        self.mask = torch.zeros([word_level.shape[0],word_level.shape[2]])
        
        ind = torch.eq(question_encoding,torch.zeros([1,self.vocab_size]).cuda())
        self.mask = torch.sum(question_encoding, dim=2) # B*Seq

        quesAtt1 = self.atten_net(image, word_level, self.mask)

        return word_level


