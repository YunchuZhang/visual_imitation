import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dir', type=str)
	parser.add_argument('--val_dir', type=str)
	parser.add_argument('--test_dir', type=str)
	parser.add_argument('--gpu', type=str, default='True')
	parser.add_argument('--save_dir', type=str, default="results")
	parser.add_argument('--exp_name', type=str, default='todo')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--pretrained', type=str, default='True')
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--feat_dim', type=int, default=256)
	parser.add_argument('--epochs', type=int, default=60)
	parser.add_argument('--model', type=str, default="Seq_gcbc", choices=["policy"])
	parser.add_argument('--env', type=str, default="trash")
	parser.add_argument('--history', type=int, default=1)
	parser.add_argument('--mult', type=int, default=0)
	parser.add_argument('--mirror', type=int, default=0)
	parser.add_argument('--rot', type=str, default="6d-d")
	parser.add_argument('--seq_len', type=int, default=5) # data_seq_len
	parser.add_argument('--data_size', '-fraction of runs to use', type=float, default=1)
	
	parser.add_argument('--l1', type=float, default=0)  # weight for l1 loss
	parser.add_argument('--l2', type=float, default=1)  # weight for l2 loss
	parser.add_argument('--l3', type=float, default=0)  # weight for direction loss
	parser.add_argument('--lg', type=float, default=0)  # weight for gripper_loss
	parser.add_argument('--lag', type=float, default=0)  # weight for angle_loss

	parser.add_argument('--rad', type=str, default="all")
	parser.add_argument('--n_head', type=int, default=8)
	parser.add_argument('--n_layers', type=int, default=1) #transformerlayer
	parser.add_argument('--trans_dim', type=int, default=2048)
	parser.add_argument('--action_dim', type=int, default=9)
	parser.add_argument('--num_dis', type=int, default=64) # num of distributions 
	parser.add_argument('--grip_file', type=str, required=False)
	parser.add_argument('--rand', type=int, default=0) # seed
	parser.add_argument('--task', type=str, required=True, choices=["push"])
	args = parser.parse_args()

	# convert to dictionary
	params = vars(args)
	return params, args
