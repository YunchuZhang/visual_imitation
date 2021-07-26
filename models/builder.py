import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as D 
from torchvision import models
import numpy as np 
import math


class Featurenet(nn.Module):
	'''
	for feature extraction
	'''
	def __init__(self, params):
		super(Featurenet, self).__init__()

		self.feat_dim = params['feat_dim']
		self.pretrained = params['pretrained']

		# 1.first resnet18 to get features with size 512
		resnet = models.resnet18(pretrained = self.pretrained)
		self.encoder = nn.Sequential(*list(resnet.children())[0:9])

		self.fc1 = nn.Linear(512, self.feat_dim)

	def forward(self, cur_img):
		'''
		input : cur_img (B,C,H,W)
		output: feature (B,E)
		'''
		x = self.fc1(torch.squeeze(self.encoder(cur_img)))
		x = F.relu(x)
		x = x.view(-1,self.feat_dim)
		return x

class PositionalEncoding(nn.Module):
	'''
	position encoding with sin and cos
	'''
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class SeqGoalBC(nn.Module):
	'''
	transformer plus goal conditioned BC
	'''
	def __init__(self, params):
		super(SeqGoalBC, self).__init__()
		self.device = torch.device("cuda:0" if (torch.cuda.is_available() and params['gpu']) else "cpu")
		self.feat_dim = params['feat_dim']
		self.tfeat_dim = params['feat_dim'] * 2
		self.n_head = params['n_head']
		self.n_layers = params['n_layers']
		self.action_dim = params['action_dim']
		self.num_dis = params['num_dis']
		self.dim = params['trans_dim']
		self.seq_len = params['seq_len']


		# sequential network (transfermer)
		# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.tfeat_dim, nhead = self.n_head, dim_feedforward=self.dim)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
		self.pos_encoder = PositionalEncoding(self.tfeat_dim, dropout = 0.1, max_len=self.seq_len)

		# feature extraction
		self.img_emb = Featurenet(params)

		# gripper net
		self.fc1 = nn.Linear(self.tfeat_dim, 256)
		self.fc2 = nn.Linear(256, 1)
		self.g_net = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.Sigmoid())

		# POSITION ANGLE NET
		self.fcw = nn.Linear(self.tfeat_dim, self.action_dim*self.num_dis)
		self.fcu = nn.Linear(self.tfeat_dim, self.action_dim*self.num_dis)
		self.fcs = nn.Linear(self.tfeat_dim, self.action_dim*self.num_dis)

		# action net
		# DEPENDENT, 6D: input 3 images + pos (3) concatenated, output angle in 6d representation
		dep_a6dfc1 = nn.Linear(in_features=512, out_features=256, bias=True)
		dep_a6dfc2 = nn.Linear(in_features=256, out_features=6, bias=True)
		self.dep_a6d_net = nn.Sequential(dep_a6dfc1, nn.ReLU(), dep_a6dfc2)

		# POSITION NET
		pfc1 = nn.Linear(in_features=512, out_features=256, bias=True)
		pfc2 = nn.Linear(in_features=256, out_features=3, bias=True)
		self.p_net = nn.Sequential(pfc1, nn.ReLU(), pfc2)


	def logistic_mixture(self, inputs, action_dim = 9, num_dis = 64):
		'''
		Args:
			similar to VAE
			use weights, mu, sig to generate mixture of distributions
			action_dim 9
			weights,u,sig B, 9, num_dis
		Output: 
			sample actions B,9
		'''
		weights, mu, scale = inputs
		# logistics distributions 
		u_min = torch.zeros(mu.shape).to(self.device)
		u_max = torch.ones(mu.shape).to(self.device)
		base_distribution = D.Uniform(u_min,u_max)
		transforms = [D.SigmoidTransform().inv, D.AffineTransform(loc=mu, scale=scale)]
		logistic = D.TransformedDistribution(base_distribution, transforms)
		
		weightings = D.Categorical(logits=weights)
		mixture_dist = D.MixtureSameFamily(
			mixture_distribution=weightings,
			component_distribution=logistic)
		return mixture_dist.sample()

	def forward(self, imgs, gripper = False):
		'''
		Args:
			imgs : (B, S+1, E) 16,3+1,512 (seq+goal)
			gripper : whether predict gripper 
		Output:
			translation : (B, 3)
			rotation : (B, 6)
			gripper : (B, 1)
		'''
		# 
		B, S, C, H, W = imgs.shape
		# get feature embeddings  
		feat = self.img_emb(imgs.reshape(B*S, C, H, W)) # (B*S, E)	
		feat = feat.reshape(B, S, self.feat_dim) 
		# concat goal feature to each feature (B, S-1, tfeat_dim) 
		feat = torch.cat([feat[:,:-1,:], feat[:,-1:,:].repeat(1,S-1,1)], dim=-1)

		feat = feat.permute(1,0,2).float() # (S, B, E)
		feat = self.pos_encoder(feat)
		feat = self.transformer_encoder(feat)

		emb = feat.permute([1,0,2]).mean(dim = 1) # B,tfeat_dim
		# gripper net
		g_pred = torch.tensor([])
		if gripper == True:
			g_pred = self.g_net(emb).reshape(B,1)
		
		angle = self.dep_a6d_net(emb).reshape(B,6)
		pos = self.p_net(emb).reshape(B,3)

		# weights = self.fcw(emb).reshape(B,self.action_dim,self.num_dis)
		# mu = self.fcu(emb).reshape(B,self.action_dim,self.num_dis)
		# # scale = torch.exp(0.5*self.fcs(emb+1e-4)).reshape(B,self.action_dim,self.num_dis)
		# scale = F.relu(self.fcs(emb+1e-4)).reshape(B,self.action_dim,self.num_dis)
		# # position and 6d rotaion
		# pred_act = self.logistic_mixture([weights,mu,scale],self.action_dim,self.num_dis)
			
		# return pred_act[:,:3], pred_act[:,3:], g_pred
		return pos, angle, g_pred
		
if __name__ == '__main__':
	src = torch.rand(10, 5, 3, 224, 224)
	params = {}
	params["pretrained"] = True
	params["feat_dim"] = 256
	params["tfeat_dim"] = 256 * 2
	params["n_head"] = 8
	params["n_layers"] = 1
	params["action_dim"] = 9
	params["num_dis"] = 64
	params['gpu'] = False
	params['seq_len'] = 5
	model = SeqGoalBC(params)
	# model.cuda()
	model(src,gripper=True)
	# transformer_encoder(src) (10,32,512)