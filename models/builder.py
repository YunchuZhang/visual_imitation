import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.transformer import TransformerForecaster
import torch.distributions as D 
from torchvision import models
import numpy as np 
import math


class Featurenet(nn.Module):
	'''
	for 1d feature extraction
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

class Embnet(nn.Module):
	'''
	for 2d feature extraction
	'''
	def __init__(self, params):
		super(Embnet, self).__init__()

		self.feat_dim = params['feat_dim']
		self.pretrained = params['pretrained']

		# 1.first resnet18 to get features with size 512
		resnet = models.resnet18(pretrained = self.pretrained)
		self.encoder = nn.Sequential(*list(resnet.children())[0:8])


	def forward(self, cur_img):
		'''
		input : cur_img (B,C,H,W)
		output: feature (B,C,7,7)
		'''
		return self.encoder(cur_img)

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
		self.tfeat_dim = params['feat_dim'] * 1
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
		self.pos_encoder = PositionalEncoding(self.tfeat_dim, dropout = 0.1, max_len=self.seq_len+1)

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
		dep_a6dfc1 = nn.Linear(in_features=self.tfeat_dim, out_features=128, bias=True)
		dep_a6dfc2 = nn.Linear(in_features=128, out_features=6, bias=True)
		self.dep_a6d_net = nn.Sequential(dep_a6dfc1, nn.ReLU(), dep_a6dfc2)

		# POSITION NET
		pfc1 = nn.Linear(in_features=self.tfeat_dim, out_features=128, bias=True)
		pfc2 = nn.Linear(in_features=128, out_features=3, bias=True)
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
		# feat = torch.cat([feat[:,:-1,:], feat[:,-1:,:].repeat(1,S-1,1)], dim=-1)
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

class Denseattention(nn.Module):
	'''
	transformer with current img and goal ima
	'''
	def __init__(self, params):
		super(Denseattention, self).__init__()
		self.device = torch.device("cuda:0" if (torch.cuda.is_available() and params['gpu']) else "cpu")
		self.feat_dim = params['feat_dim']
		self.tfeat_dim = params['feat_dim'] * 1
		self.n_head = params['n_head']
		self.n_layers = params['n_layers']
		self.action_dim = params['action_dim']
		self.num_dis = params['num_dis']
		self.dim = params['trans_dim']
		self.seq_len = params['seq_len']


		self.transformer_encoder = TransformerForecaster(seqlen=self.seq_len)
		# feature extraction
		self.img_emb = Embnet(params)

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
		dep_a6dfc1 = nn.Linear(in_features=self.feat_dim, out_features=256, bias=True)
		dep_a6dfc2 = nn.Linear(in_features=256, out_features=6, bias=True)
		self.dep_a6d_net = nn.Sequential(dep_a6dfc1, nn.ReLU(), dep_a6dfc2)

		# POSITION NET
		pfc1 = nn.Linear(in_features=self.feat_dim, out_features=256, bias=True)
		pfc2 = nn.Linear(in_features=256, out_features=3, bias=True)
		self.p_net = nn.Sequential(pfc1, nn.ReLU(), pfc2)


	def forward(self, imgs, gripper = False):
		'''
		Args:
			imgs : (B, S+1, E) 16,1+1,512,7,7 (seq+goal)
			gripper : whether predict gripper 
		Output:
			translation : (B, 3)
			rotation : (B, 6)
			gripper : (B, 1)
		'''
		# 
		B, S, C, H, W = imgs.shape
		# get feature embeddings  
		feat = self.img_emb(imgs.reshape(B*S, C, H, W)) # (B*S, E, H, W)  
		_,_,h,w = feat.shape
		feat = feat.reshape(B, S, self.feat_dim, h,w)

		pes = self.pos_encoder(
		torch.zeros(B,S,h,w).to(self.device),
		torch.zeros(B,S,h,w).to(self.device),
		torch.zeros(B,S,h,w).to(self.device),
		S, h, w, int(np.ceil(self.feat_dim/3))) # B, C, S, h, w

		pes = pes.permute(0, 2, 1, 3, 4) # B, S, C, h, w
		emb = self.transformer_encoder(feat,pes[:,:,:512]) # B, 3+S*(H*W), C

		# gripper net
		g_pred = torch.tensor([])
		if gripper == True:
			g_pred = self.g_net(torch.squeeze(emb[:,2,:])).reshape(B,1)
		
		angle = self.dep_a6d_net(torch.squeeze(emb[:,1,:])).reshape(B,6)
		pos = self.p_net(torch.squeeze(emb[:,0,:])).reshape(B,3)

		# weights = self.fcw(emb).reshape(B,self.action_dim,self.num_dis)
		# mu = self.fcu(emb).reshape(B,self.action_dim,self.num_dis)
		# # scale = torch.exp(0.5*self.fcs(emb+1e-4)).reshape(B,self.action_dim,self.num_dis)
		# scale = F.relu(self.fcs(emb+1e-4)).reshape(B,self.action_dim,self.num_dis)
		# # position and 6d rotaion
		# pred_act = self.logistic_mixture([weights,mu,scale],self.action_dim,self.num_dis)
			
		# return pred_act[:,:3], pred_act[:,3:], g_pred
		return pos, angle, g_pred

	def pos_encoder(self, z, y, x, Z, Y, X, C ):
		B = x.shape[0]
		# returns pe shaped B, C*3, Z, Y, X, where the C has the sincos for x,y,z
		grid_z, grid_y, grid_x = self.meshgrid3d(B, Z, Y, X) # B, H, W
		grid_z = (grid_z - 0).unsqueeze(1) # B, 1, Z, Y, X
		grid_y = (grid_y - y).unsqueeze(1) # B, 1, Z, Y, X
		grid_x = (grid_x - x).unsqueeze(1) # B, 1, Z, Y, X

		div_term = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C)).reshape(1, int(np.ceil(C/2)), 1, 1, 1).to(x.device)
		
		pe_z = torch.zeros(B, int(np.ceil(C/2))*2, Z, Y, X).to(x.device)
		pe_y = torch.zeros(B, int(np.ceil(C/2))*2, Z, Y, X).to(x.device)
		pe_x = torch.zeros(B, int(np.ceil(C/2))*2, Z, Y, X).to(x.device)
		
		pe_z[:, 0::2] = torch.sin(grid_z * div_term)
		pe_z[:, 1::2] = torch.cos(grid_z * div_term)

		pe_y[:, 0::2] = torch.sin(grid_y * div_term)
		pe_y[:, 1::2] = torch.cos(grid_y * div_term)
		
		pe_x[:, 0::2] = torch.sin(grid_x * div_term)
		pe_x[:, 1::2] = torch.cos(grid_x * div_term)

		# print('pe_x', pe_x.shape)
		# print('pe_y', pe_y.shape)
		# print('pe_z', pe_z.shape)
		
		pe = torch.cat([pe_z, pe_y, pe_x], dim=1)
		return pe
	
	def meshgrid3d(self, B, Z, Y, X, stack=False, norm=False):
		# returns a meshgrid sized B x Z x Y x X
		
		grid_z = torch.linspace(0.0, Z-1, Z)
		grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
		grid_z = grid_z.repeat(B, 1, Y, X)

		grid_y = torch.linspace(0.0, Y-1, Y)
		grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
		grid_y = grid_y.repeat(B, Z, 1, X)

		grid_x = torch.linspace(0.0, X-1, X)
		grid_x = torch.reshape(grid_x, [1, 1, 1, X])
		grid_x = grid_x.repeat(B, Z, Y, 1)

		# if cuda:
		#     grid_z = grid_z.cuda()
		#     grid_y = grid_y.cuda()
		#     grid_x = grid_x.cuda()

		grid_z = grid_z.to(self.device)
		grid_y = grid_y.to(self.device)
		grid_x = grid_x.to(self.device)
			
		if norm:
			grid_z, grid_y, grid_x = normalize_grid3d(
				grid_z, grid_y, grid_x, Z, Y, X)

		if stack:
			# note we stack in xyz order
			# (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
			grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
			return grid
		else:
			return grid_z, grid_y, grid_x

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
if __name__ == '__main__':
	src = torch.rand(10, 2, 3, 224, 224)
	params = {}
	params["pretrained"] = True
	params['trans_dim'] = 2048
	params["feat_dim"] = 512
	params["tfeat_dim"] = 256 * 2
	params["n_head"] = 8
	params["n_layers"] = 1
	params["action_dim"] = 9
	params["num_dis"] = 64
	params['gpu'] = False
	params['seq_len'] = 5
	model = SeqGoalBC(params)
	model = Denseattention(params)
	# model.cuda()
	model(src,gripper=False)
	# transformer_encoder(src) (10,32,512)