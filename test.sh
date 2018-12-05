#!/usr/bin/env bash

python main.py \
--train_data_dir /home/dwt/DataServer/UserdirBackup/cym/train \
--eval_data_dir /home/dwt/DataServer/UserdirBackup/cym/train \
--train_list_file train.txt \
--eval_list_file eval.txt \
--batch_size 4 \
--epoches 10 \
--lr=1e-2 \
--momentum 0.9 \
--weight_decay 1e-4 \
--seed 0 \
--log_interval 1 \
--snapshot_interval 1 \
--snapshot_prefix ./snapshot/model \
--gpu 2 \
--weights ./pretrained/vgg_SCNN_DULR_w9.pth
#--test_data_dir /home/dwt/DataServer/UserdirBackup/cym/test \
#--test_list_file test0_normal.txt
#--checkpoint ./snapshot/lr0.01_pretrained_model_batch_2359.pth
