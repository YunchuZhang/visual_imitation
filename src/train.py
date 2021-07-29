import torch
import json
import glob
import time
from args import get_args
import sys,os
sys.path.append(os.getcwd())

from models.builder import SeqGoalBC
from trainer import PolicyTrainer
from utils.get_result_vids import makeVideo


def main(params, args):

	# logdir
	logdir_prefix = 'policy_' + args.exp_name

	data_path = os.path.join(os.getcwd(), args.save_dir)

	if not (os.path.exists(data_path)):
		os.makedirs(data_path)

	logdir = logdir_prefix + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
	logdir = os.path.join(data_path, logdir)
	params['logdir'] = logdir
	if not (os.path.exists(logdir)):
		os.makedirs(logdir)
		os.makedirs(logdir + "/valimages/")
		os.makedirs(logdir + "/trainimages/")
		os.makedirs(logdir + "/testimages/")

	print("\n\nLOGGING TO: ", logdir, "\n\n")
	device = torch.device('cuda:0' if (torch.cuda.is_available() and params['gpu']) else "cpu")

	print("Training a " + params['model'] + "\n\n")

	model = SeqGoalBC(params).to(device)

	with open(logdir + '/params.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, separators=(',', ': '), sort_keys=True)
		# add trailing newline for POSIX compatibility
		outfile.write('\n')

	trainer = PolicyTrainer(
			model,
			params
		)
	
	trainer.train(logdir)

	makeVideo(logdir, logdir, "test")
	makeVideo(logdir, logdir, "val")
	makeVideo(logdir, logdir, "train")

	vids = sorted(glob.glob(logdir + "/*.avi"))
	for ff in vids:
		os.system("ffmpeg -i " + ff + " " + ff[:-4] + ".mp4")
		os.remove(ff)
	print(params)

if __name__ == '__main__':
	params, args = get_args()
	main(params, args)