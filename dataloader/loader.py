# Customized from Visual Imitation made it easy
from __future__ import print_function, division
import glob
import numpy as np 
import torch  
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial.transform import Rotation as R  
import random   
import json
import os


class colmapdataset(Dataset):
	"""dataset without gripper motion"""
	def __init__(self, dataset, params, transform = None, cache_size=500):
		"""
		dataset : train/test/val dataset path with images
		params : useful params 
		transform : Optional transform provided by user/ otherwise use default one
		"""
		if dataset == "train":
			root_dir = params['train_dir']
		elif dataset == "val":
			root_dir = params['val_dir']
		elif dataset == "test":
			root_dir = params['test_dir']
		self.mirror = params['mirror']
		self.rad = params['rad']
		folders = sorted(glob.glob(root_dir + "/*"), reverse=True)
		print("dir", root_dir + "/*")
		self.data_size = params['data_size']
		self.seq_len = params['seq_len']
		self.dataset = dataset

		self.train = []
		if transform != None:
			self.transform = transform
		else:
			if self.dataset != "train" or not params['rad'] or params['rad'] == "None":
				self.transform = transforms.Compose([  # [1]sc
					transforms.Resize(240),  # [2]
					transforms.CenterCrop(224),  # [3]
					transforms.ToTensor(),  # [4]
				])

				self.h_transform = transforms.Compose([  # [1]
					transforms.Resize(240),  # [2]
					transforms.CenterCrop(224),  # [3]
					transforms.RandomHorizontalFlip(1),
					transforms.ToTensor(),  # [4]
				])
			else:

				t_lst = [transforms.Resize(240)]

				if "rotate" in self.rad or self.rad == "all":
					t_lst.append(transforms.RandomRotation(5))

				if "crop" in self.rad or self.rad == "all":
					t_lst.append(transforms.RandomCrop(224))
				else:
					t_lst.append(transforms.CenterCrop(224))

				if "jitter" in self.rad or self.rad == "all":
					t_lst.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.2))

				rand_val = random.random()
				if "mirror" in self.rad and rand_val < 0.5:
					t_lst.append(transforms.RandomHorizontalFlip(1))

				print("List of transforms:", t_lst + [transforms.ToTensor()])

				self.transform = transforms.Compose(t_lst + [transforms.ToTensor()])
				self.h_transform = transforms.Compose(t_lst + [transforms.RandomHorizontalFlip(1), transforms.ToTensor()])


		total_imgs = 0
		total_runs = len(folders)
		self.all_translations = []

		counter = 0
		used_runs = 0
		target_runs = round(total_runs * self.data_size)
		for run in folders:
			
			counter += 1
			if self.dataset == "train":
				if self.data_size == 0.01 and counter % 100 != 0:
					continue
				if self.data_size == 0.05 and counter % 20 != 0:
					continue
				if self.data_size == 0.1 and counter % 10 != 0:
					continue
				if self.data_size == 0.25 and counter % 4 != 0:
					continue
				if self.data_size == 0.5 and counter % 2 != 0:
					continue
				if self.data_size == 0.75 and counter % 4 == 3:
					continue
			else:
				if self.dataset == "train":
					if used_runs > target_runs:
						continue


			# Only gets images, goals, trans, rots that actually exist.
			the_old_imgs, translations, rotations = self.get_vals(run)
			
			imgs = the_old_imgs
			if len(the_old_imgs) == 0 or len(translations) == 0:
				continue

			# scale factor that divides all actions by max(x,y,z) across a run, for every run.
			scale_factor = np.max(np.abs(translations))
			if scale_factor >= 2.5 or scale_factor < 0.05:
				print("Max norm(action) is not normal for run " + run + ": ", scale_factor)
				continue

			used_runs += 1
			scaled_trans = translations / scale_factor

			num_imgs = len(imgs)
			total_imgs += num_imgs

			for i in range(num_imgs):
				img_t = imgs[i]

				the_xyz = scaled_trans[i]
				# if self.normed == 1:
				# 	# divide by L1 norm for each action to get a direction.
				# 	the_xyz = the_xyz / np.linalg.norm(scaled_trans[i], 1)

				self.train.append(([img_t], np.vstack((the_xyz, rotations[i])), run, 0)) # not mirrored

				if self.mirror:
					negged = the_xyz
					negged[0] = -negged[0]
					r = R.from_dcm(rotations[i])
					euler = r.as_euler('xyz')
					euler[0] *= -1
					flipped = R.from_euler('xyz', euler).as_dcm()
					self.train.append(([img_t], np.vstack((negged, flipped)), run, 1)) # mirrored


		print("num used runs:", used_runs)
		print("num collected images: ", total_imgs)
		if self.mirror:
			print("Augmented images with mirror:", total_imgs * 2)

		self.cache_size = cache_size # how many data points to cache in memory
		self.cache = {} # from index to tuple

	def get_vals(self, f):
		input_file = open(f + '/labels.json', 'r')
		json_decode = json.load(input_file)
		trans = []
		rot = []
		imgs = []
		items = sorted(json_decode.items(), key=lambda x: x[0])
		goal_img = "theframe{:04d}.png".format(int(items[-1][0][8:-4])+1)
		if not os.path.exists(f + "/images/" + goal_img):
			print("lossing goal_img of {}".format(f + "/images/" + goal_img))
			return [],[],[]
		assert os.path.exists(f + "/images/" + goal_img), "lossing goal_img of {}".format(f + "/images/" + goal_img)
		for img, vals in items:
			if os.path.exists(f + "/images/" + img):
				t, r = vals
				# if max(np.abs(np.array(t))) < 0.05:
				# 	continue
				trans.append(t)
				rot.append(r)
				imgs.append(f + "/images/" + img)
		rot = np.array(rot)
		trans = np.array(trans)
		imgs = np.array(imgs)
		# split data to sequence data format
		split_imgs, split_trans, split_rot = self.split_data(imgs,trans,rot,np.array([f + "/images/" + goal_img]))
		return split_imgs, split_trans, split_rot 

	def split_data(self, imgs, trans, rot, goal_img, stride = 1):
		'''
		sliding window with lengh and stride
		also append goal img path
		'''

		# stride = int(self.seq_len//2)
		split_imgs = []
		split_trans = []
		split_rot = []
		for n in range(0, len(imgs) - self.seq_len, stride):
			split_imgs.append(np.concatenate((imgs[n:n+self.seq_len],goal_img)))
			split_trans.append(trans[n+self.seq_len-1])
			split_rot.append(rot[n+self.seq_len-1])
		
		return split_imgs, split_trans, split_rot

	def __len__(self):
		return len(self.train)

	def __getitem__(self, idx):
		# return imgs with goal
		if idx in self.cache:
			img_t, imgs_withgoal, imgs_goal, actions, run, suffix = self.cache[idx]
		else:
			img_names, actions, run, mirrored = self.train[idx]
			img_t = img_names[0][0]
			goal_img = img_names[0][-1]
			imgs = []
			if mirrored:
				for i in range(self.seq_len+1):
					imgs.append(torch.unsqueeze(self.transform(Image.open(img_names[0][i])), 0))
				imgs_goal = torch.unsqueeze(self.h_transform(Image.open(goal_img)), 0)
				img_t = img_t[:-4] + "_mirrored" + img_t[-4:]
			else:
				for i in range(self.seq_len+1):
					imgs.append(torch.unsqueeze(self.transform(Image.open(img_names[0][i])), 0))
				imgs_goal = torch.unsqueeze(self.transform(Image.open(goal_img)), 0)

			if self.dataset == "train" and "cut" in self.rad:
				opened = self.random_cutout_color(imgs.numpy(), min_cut=10, max_cut=210)
				imgs = opened
			suffix = []
			imgs_withgoal = torch.cat(imgs,dim=0)
			if len(self.cache) < self.cache_size:
				self.cache[idx] = (img_t, imgs_withgoal, imgs_goal, actions, run, suffix)
		return img_t, imgs_withgoal, imgs_goal, actions, run, suffix

	# From RAD
	def random_cutout_color(self, imgs, min_cut=10, max_cut=30, min_w=10, max_w=60):
		"""
				args:
				imgs: shape (B,C,H,W)
				out: output size (e.g. 84)
		"""

		n, c, h, w = imgs.shape
		w1 = np.random.randint(min_cut, max_cut, n)
		h1 = np.random.randint(min_cut, max_cut, n)
		boxw = np.random.randint(min_w, max_w, n)
		boxh = np.random.randint(min_w, max_w, n)
		cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)

		rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
		for i, (img, w11, h11, boxww, boxhh) in enumerate(zip(imgs, w1, h1, boxw, boxh)):
			cut_img = img.copy()
			# add random box
			cut_img[:, h11:h11 + boxhh, w11:w11 + boxww] = np.tile(
				rand_box[i].reshape(-1, 1, 1),
				(1,) + cut_img[:, h11:h11 + boxhh, w11:w11 + boxww].shape[1:])

			cutouts[i] = cut_img
		return cutouts

if __name__ == '__main__':
	params = {}
	params['train_dir'] = 'data/train'
	params['mirror'] = 0
	params['rad'] = 'all'
	params['seq_len'] = 5
	params['data_size'] = 1
	d = colmapdataset("train",params)
	import ipdb;ipdb.set_trace()
	d[0]
	# should not shuffle? since depends on time sequence
	train_loader = DataLoader(d, batch_size=16, shuffle=True,
													sampler=None,
													batch_sampler=None, num_workers=4,
													pin_memory=False, drop_last=False, timeout=0,
													worker_init_fn=None)
	for i_batch, (batch_img_names, batch_imgs, batch_imgs_goal, batch_actions, runs, aug) in enumerate(train_loader):
		print(batch_img_names)
		batch_imgs = batch_imgs.to("cpu")