#!/bin/bash
export WANDB_PROJECT="restore"
export WANDB_NAME="run_name" # change to your counts

MODEL_DIR="checkpoints/model"
OUTPUT_DIR="checkpoints/out"

export SWIFT_PATCH_CONV3D=1

TRAIN_DATASET="train.json"
VAL_DATASET="val.json"

# might try a different acc_strategy some time, also probably increase max_len later, 99% of current dataset falls under that len though

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model "$MODEL_DIR" \
    --model_type qwen3_vl \
    --template qwen3_vl \
    --dataset ${TRAIN_DATASET} \
    --val_dataset ${VAL_DATASET} \
    --train_type full \
    --use_liger true \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --max_steps 12000 \
    --learning_rate 8e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --save_steps 100 \
    --acc_strategy token \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --max_length 3000 \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --report_to wandb \
    --ddp_backend nccl \
    --ddp_find_unused_parameters true
