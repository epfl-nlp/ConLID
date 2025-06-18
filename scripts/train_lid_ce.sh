#!/bin/bash

# Multi-GPU run
WORLD_SIZE=${WORLD_SIZE:-1}
WORLD_SIZE=1
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
LR=$(python -c "print(0.001 * int($N_GPUS))")

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    $REPO_PATH/conlid/train/train.py \
    --model_type lid_ce \
    --data_dir $REPO_PATH/data/glotlid/v3.1 \
    --config_path $REPO_PATH/conlid/config.json \
    --run_name LID-CE \
    --output_dir $REPO_PATH/checkpoints/lid_ce \
    --eval_strategy steps \
    --eval_steps 0.05 \
    --eval_on_start True \
    --logging_strategy steps \
    --logging_steps 0.01 \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 2 \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --learning_rate $LR \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --torch_empty_cache_steps 256 \
    --num_train_epochs 1 \
    --push_to_hub false \
    --remove_unused_columns false \
    --report_to wandb \
    --seed 42