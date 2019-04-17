import torch.nn as nn
import torch.nn.functional as F
import torch
from external.resnet.resnet import resnet18


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
		self.x_embedding = nn.Linear(self.embed_size, self.hidden_size)
		self.g_embedding = nn.Linear(self.embed_size, self.hidden_size)

		self.hx_weights = nn.Linear(self.hidden_size,1)
		self.softmax = nn.Softmax()
		self.masksoftmax = MaskSoftMax()

		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(0.5)

	def forward(self,image_feat, ques_feat, mask):
		# Three steps:
		# Ques_feat size D*T
		# Image_feat size D*N e.g. D*196
		# First step:
		feat = self.x_embedding(ques_feat)
		feat = self.tanh(feat)
		feat = self.dropout(feat)
		#feat: K*T
		h1 = self.hx_weights(feat)
		#h1: 1*T
		h1 = torch.squeeze(h1,dim=2) #B*Seq
		p1 = self.masksoftmax(h1, mask) # B*Seq
		#p1: 1*T
		ques_atten = torch.unsqueeze(p1, dim=1) 
		quesAtt1 = torch.matmul(ques_atten,ques_feat)
		# quesAtt1 = torch.squeeze(quesAtt1,dim=1)
	 	#quesAtt1: B*1*Embed

	 	# Second step:
		x_feat = self.x_embedding(image_feat) # B*N*Hidden
		g_feat = self.g_embedding(quesAtt1) # B*1*Hidden

		added = x_feat + g_feat # B*N*Hidden

		added = self.tanh(added)
		added = self.dropout(added)

		h2 = self.hx_weights(added)
		p2 = self.softmax(h2) # B*N*1

		img_atten = p2.permute(0,2,1) 
		visAtt = torch.matmul(img_atten, image_feat) # image attention feature v hat
		# visAtt: B*1*Embed
		# Thrid step
		img_embed = self.g_embedding(visAtt)
		ques_embed_dim = self.x_embedding(ques_feat)

		added = img_embed + ques_embed_dim
		added = self.tanh(added)
		added = self.dropout(added) # B*Seq*Emb
		h3 = self.hx_weights(added) # B*Seq*1
		h3 = torch.squeeze(h3,dim=2)
		p3 = self.masksoftmax(h3, mask)
		ques_atten = torch.unsqueeze(p1, dim=1) 
		quesAtt = torch.matmul(ques_atten,ques_feat) # question attention feature
		# quesAtt: B*1*Embed
		return quesAtt, visAtt

class ResNet(nn.Module):
	def __init__(self):
		super(ResNet,self).__init__()
		self.upsample = nn.Upsample((448,448), mode='bilinear')
		self.model = resnet18(pretrained=True)
		def save_output(module, intput, output):
			self.buffer = output
		self.model.layer4.register_forward_hook(save_output)

	def forward(self,x):
		x = self.upsample(x)
		x = self.model(x)
		return self.buffer

	def freeze(self):
		for param in self.model.parameters():
			param.requires_grad = False

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, embed_size, vocab_size, ans_size, seq_len, hidden_size=512):
        super().__init__()



        self.hidden_size = hidden_size
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
        self.softmax = nn.Softmax()

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
        self.image_feat = ResNet()
        self.image_feat.freeze()
        self.atten_net = AttentionNet(self.embed_size, self.seq_length, self.embed_size)

        self.word_linear = nn.Linear(self.embed_size, self.hidden_size)
        self.phrase_linear= nn.Linear(self.embed_size+self.hidden_size, self.hidden_size)
        self.ques_linear = nn.Linear(self.embed_size+self.hidden_size, self.hidden_size)
        self.out_linear = nn.Linear(self.hidden_size, self.ans_size)

    def forward(self, image, question_encoding, prepro=False):
        # TODO

        # question_encoding: B*Seq*vocab_size
        if prepro==False:
        	image = self.image_feat(image)

        image = image.view(-1,512,196)
        image = image.permute(0,2,1)

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

        quesAtt_word, visAtt_word = self.atten_net(image, word_level, self.mask)
        quesAtt_phrase, visAtt_phrase = self.atten_net(image, phrase_level, self.mask)
        quesAtt_ques, visAtt_ques = self.atten_net(image, ques_level, self.mask)
        
        # Encoded answers
        feat1 = quesAtt_word + visAtt_word
        feat1 = self.dropout(feat1)
        hidden1 = self.word_linear(feat1)
        hidden1 = self.tanh(hidden1)
        feat2 = quesAtt_phrase + visAtt_phrase
        feat2 = torch.cat((feat2,hidden1),dim=2)
        feat2 = self.dropout(feat2)
        hidden2 = self.phrase_linear(feat2)
        hidden2 = self.tanh(hidden2)
        feat3 = quesAtt_ques + visAtt_ques
        feat3 = torch.cat((feat3,hidden2),dim=2)
        feat3 = self.dropout(feat3)
        hidden3 = self.ques_linear(feat3)
        tmp = self.dropout(hidden3)
        outfeat = self.out_linear(tmp)
        outfeat = torch.squeeze(outfeat,dim=1)
        ansfeat = self.softmax(outfeat)

        return ansfeat


