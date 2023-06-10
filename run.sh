python run.py --train_dataset_dir '/media/saket/Elements/datasets/handpalm/hand_seg/egohos_dataset/test_outdomain' \
                --val_dataset_dir '/media/saket/Elements/datasets/handpalm/hand_seg/egohos_dataset/test_outdomain/' \
                --epochs 100 \
                --start_epoch 0 \
		        --lr 0.00005 \
                --batch_size 1 \
                --dataset_name 'EgoHands' \
                --optimizer_name 'Adam' \
                --ex_name ex2 \
		        --restore_from checkpoints/ex1/40/ckpt
