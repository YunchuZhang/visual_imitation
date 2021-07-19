import torch.nn as nn
from torchvision import models
import numpy as np 


class GoalBC(nn.Module):
	def __init__(self, paras):
		super(GoalBC, self).__init__()
		resnet = models.resnet18(pretrained = pretrained)
		self.encoder = nn.Sequential(*list(resnet.children())[0:8])

		self.fc1 = nn.Linear(512, feat_dim)
			
	def forward(self, x, goal):
		x = self.fc1(self.encoder(x))
		#x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,256)
		goal = self.fc1(self.encoder(goal))
		#x = self.pool(F.relu(self.conv2(x)))
		goal = goal.view(-1,256)
		return 
sadfdfd