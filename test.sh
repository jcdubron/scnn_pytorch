#!/usr/bin/env bash

python ./python/main.py \
--data_dir /var/nfs-dir/UserdirBackup/cym \
--train_list_file train.txt \
--eval_list_file eval.txt \
--batch_size 4 \
--epoches 10 \
--lr=1e-4 \
--momentum 0.9 \
--weight_decay 1e-4 \
--seed 16 \
--log_interval 1 \
--snapshot_interval 1 \
--snapshot_prefix ./snapshot/model \
--weights VGG16.pth \
--gpu 0 3