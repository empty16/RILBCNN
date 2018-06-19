#!/usr/bin/env bash
# utilities

CUDA_VISIBLE_DEVICES=6 bash ./scripts/Train_Outex_ori_aug.sh
CUDA_VISIBLE_DEVICES=5 bash ./scripts/Train_Outex_rot_aug.sh
CUDA_VISIBLE_DEVICES=6 bash ./scripts/Train_Outex_ori2rot_aug.sh
# to load the big dataset, large memory is needed