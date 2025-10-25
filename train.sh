#!/bin/bash

python main.py \
    --dataset eth5 \
    --test_set eth \
	--reproducibility True \
	--phase 'train_test' \
	--num_epochs 250 \
	--batch_size 64 \
	--start_validation 1 \
	--validate_every 10 \
	--learning_rate 0.001 \
	--skip_ts_window 1 \
	--down_factor 8 \
	--num_workers 2 \
	--use_wandb True


# python main.py \
#     --dataset eth5 \
#     --test_set hotel \
# 	--reproducibility True \
# 	--phase 'train_test' \
# 	--num_epochs 250 \
# 	--batch_size 64 \
# 	--start_validation 1 \
# 	--validate_every 10 \
# 	--learning_rate 0.001 \
# 	--skip_ts_window 1 \
# 	--down_factor 8 \
# 	--num_workers 2 \
# 	--use_wandb True


# python main.py \
#     --dataset eth5 \
#     --test_set univ \
# 	--reproducibility True \
# 	--phase 'train_test' \
# 	--num_epochs 250 \
# 	--batch_size 64 \
# 	--start_validation 1 \
# 	--validate_every 10 \
# 	--learning_rate 0.001 \
# 	--skip_ts_window 1 \
# 	--down_factor 8 \
# 	--num_workers 2 \
# 	--use_wandb True


# python main.py \
#     --dataset eth5 \
#     --test_set zara1 \
# 	--reproducibility True \
# 	--phase 'train_test' \
# 	--num_epochs 250 \
# 	--batch_size 64 \
# 	--start_validation 1 \
# 	--validate_every 10 \
# 	--learning_rate 0.001 \
# 	--skip_ts_window 1 \
# 	--down_factor 8 \
# 	--num_workers 2 \
# 	--use_wandb True


# python main.py \
#     --dataset eth5 \
#     --test_set zara2 \
# 	--reproducibility True \
# 	--phase 'train_test' \
# 	--num_epochs 250 \
# 	--batch_size 64 \
# 	--start_validation 1 \
# 	--validate_every 10 \
# 	--learning_rate 0.001 \
# 	--skip_ts_window 1 \
# 	--down_factor 8 \
# 	--num_workers 2 \
# 	--use_wandb True