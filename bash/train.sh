python src/train.py --task push --train_dir data/pushdata/train --val_dir data/pushdata/val --test_dir data/pushdata/test --save_dir results/ --exp_name no_angle_loss

python src/train.py --task push --train_dir data/pushdata/train --val_dir data/pushdata/val --test_dir data/pushdata/test --save_dir results/ --exp_name denseattention --feat_dim 512 --seq_len 1