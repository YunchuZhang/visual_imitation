import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
import numpy as np 


class Featurenet(nn.Module):
	'''
	for feature extraction
	'''
	def __init__(self, params):
		super(Featurenet, self).__init__()

		self.feat_dim = params['feat_dim']
		self.pretrained = params['pretrained']

		# 1.first resnet18 to get features with size 512
		resnet = models.resnet18(pretrained = pretrained)
		self.encoder = nn.Sequential(*list(resnet.children())[0:9])

		self.fc1 = nn.Linear(512, self.feat_dim)

	def forward(self, cur_img):
		x = self.fc1(self.encoder(cur_img))
		x = F.relu(x)
		x = x.view(-1,self.feat_dim)
		return x

class PositionalEncoding(nn.Module):

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

		self.feat_dim = params['feat_dim'] * 2
		self.n_head = params['n_head']
		self.n_layers = params['n_layers']

		# sequential network
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.feat_dim, n_head = 8)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
		self.pos_encoder = PositionalEncoding(self.feat_dim, dropout = 0.1)

		"""
		to do 
		"""
		resnet = models.resnet18(pretrained = pretrained)
		self.encoder = nn.Sequential(*list(resnet.children())[0:9])

		self.fc1 = nn.Linear(512, feat_dim)

	def forward(self, cur_img, goal_img):
		x = self.fc1(self.encoder(cur_img))
		#x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,256)
		goal = self.fc1(self.encoder(goal))
		#x = self.pool(F.relu(self.conv2(x)))
		goal = goal.view(-1,256)
		return 



		# POSITION NET
		pfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim, bias=True).to(self.device) # *3 for 3 images
		pfc2 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)
		self.p_net = nn.Sequential(pfc1, nn.ReLU(), pfc2).to(self.device)

		if self.net_type=="alex":
			pfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim * 2, bias=True).to(self.device)  # *3 for 3 images
			pfc2 = nn.Linear(in_features=self.feat_dim * 2, out_features=self.feat_dim, bias=True).to(self.device)
			pfc3 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)  # *3 for 3 images

			self.p_net = nn.Sequential(pfc1, nn.ReLU(), pfc2, nn.ReLU(), pfc3).to(self.device)



		# ANGLE NETS

		# INDEPENDENT, 6D: input 3 images + pos (3) concatenated, output angle in 6d representation
		in_a6dfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
		in_a6dfc2 = nn.Linear(in_features=self.feat_dim, out_features=6, bias=True).to(self.device)  # 2 * 3 matrix
		self.in_a6d_net = nn.Sequential(in_a6dfc1, nn.ReLU(), in_a6dfc2).to(self.device).to(self.device)

		# INDEPENDENT, 3D: input 3 images + pos (3) concatenated, output angle in angle-axis representation
		in_a3dfc1 = nn.Linear(in_features=self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
		in_a3dfc2 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)  # 1 * 3 matrix
		self.in_a3d_net = nn.Sequential(in_a3dfc1, nn.ReLU(), in_a3dfc2).to(self.device).to(self.device)

		# DEPENDENT, 6D: input 3 images + pos (3) concatenated, output angle in 6d representation
		dep_a6dfc1 = nn.Linear(in_features=3 + self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
		dep_a6dfc2 = nn.Linear(in_features=self.feat_dim, out_features=6, bias=True).to(self.device) # 2 * 3 matrix
		self.dep_a6d_net = nn.Sequential(dep_a6dfc1, nn.ReLU(), dep_a6dfc2).to(self.device).to(self.device)

		# DEPENDENT, 3D: input 3 images + pos (3) concatenated, output angle in angle-axis representation
		dep_a3dfc1 = nn.Linear(in_features=3 + self.dimm, out_features=self.feat_dim, bias=True).to(self.device)
		dep_a3dfc2 = nn.Linear(in_features=self.feat_dim, out_features=3, bias=True).to(self.device)  # 1 * 3 matrix
		self.dep_a3d_net = nn.Sequential(dep_a3dfc1, nn.ReLU(), dep_a3dfc2).to(self.device).to(self.device)

    def __init__(self, feat_dim = 2048, pretrained = True):
        super(Conditional_Net_RN5N, self).__init__()
        resnet = models.resnet18(pretrained = pretrained)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv1 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = True)
        #self.conv2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1, bias = True)
        self.fc1 = nn.Linear(512, feat_dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
            
    def forward(self, x):
        x=self.pool(self.resnet_l5(x))
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,512)
        x = self.fc1(x)
        return x