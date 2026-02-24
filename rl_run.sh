#!/bin/bash

export IMAGE_MAX_TOKEN_NUM=4096
export WANDB_PROJECT=""
export WANDB_NAME=""

MODEL_DIR="checkpoints/in"
OUTPUT_DIR="checkpoints/out"
PLUGIN_PATH="plugins/your_reward.py"
TRAIN_DATASET="your_data.jsonl"

export SWIFT_PATCH_CONV3D=1

# GSPO settings
# --epsilon 3e-4 \
# --epsilon_high 4e-4 \
# --steps_per_generation 1 \
# --importance_sampling_level sequence \

# GDPO settings
# --scale_rewards gdpo \

# CHORD settings
# --chord_sft_dataset "$TRAIN_DATASET" \
# --chord_sft_per_device_train_batch_size 4 \
# --chord_mu_peak 0.5 \
# --chord_mu_valley 0.05 \
# --chord_mu_warmup_steps 100 \
# --chord_mu_decay_steps 2000 \
# --chord_enable_phi_function true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
  --rlhf_type grpo \
  --beta 0.001 \
  --model "$MODEL_DIR" \
  --model_type qwen3_vl \
  --template qwen3_vl \
  --dataset "$TRAIN_DATASET" \
  --train_type full \
  --freeze_vit false \
  --freeze_aligner false \
  --freeze_llm false \
  --columns '{"query": "query", "response": "solution", "images": "images"}' \
  --external_plugins "$PLUGIN_PATH" \
  --reward_funcs "YOUR REWARDS HERE" \
  --num_generations 8 \
  --max_completion_length 3000 \
  --max_steps 5000 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --save_steps 100 \
  --temperature 0.7 \
  --top_p 0.95 \
  --acc_strategy token \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_length 3000 \
  --torch_dtype bfloat16 \
  --output_dir "$OUTPUT_DIR" \
  --report_to wandb \
  --ddp_backend nccl \
  --ddp_find_unused_parameters true \
  --sleep_level 1
