#!/usr/bin/env sh

python main.py \
--train_data_dir /home/dwt/DataServer/UserdirBackup/cym/train \
--eval_data_dir /home/dwt/DataServer/UserdirBackup/cym/train \
--train_list_file train.txt \
--eval_list_file eval.txt \
--batch_size 8 \
--epoches 10 \
--batches 60000 \
--lr 1e-2 \
--momentum 0.9 \
--weight_decay 1e-4 \
--bce_weight 0.1 \
--bg_weight 0.2 \
--seed 0 \
--log_interval 25 \
--snapshot_interval 1 \
--snapshot_prefix ./snapshot/VGG_w2 \
--gpu 3 \
--weights VGG16.pth \
# --checkpoint ./snapshot/VGG_w2w2l1_batch_20000.pth \
# --test_data_dir /home/dwt/DataServer/UserdirBackup/cym/test \
# --test_log test_result.txt \
# --test_list_file ./test_list/test0_normal.txt \
#                  ./test_list/test1_crowd.txt \
#                  ./test_list/test2_hlight.txt \
#                  ./test_list/test3_shadow.txt \
#                  ./test_list/test4_noline.txt \
#                  ./test_list/test5_arrow.txt \
#                  ./test_list/test6_curve.txt \
#                  ./test_list/test7_cross.txt \
#                  ./test_list/test8_night.txt \

# --checkpoint ./snapshot/ \
# test0_normal 9621
# test1_crowd  8113
# test2_hlight 486
# test3_shadow 930
# test4_noline 4067
# test5_arrow  890
# test6_curve  422
# test7_cross  3122
# test8_night  7029
