# CONFIG

export SWIFT_PATCH_CONV3D=1
export IMAGE_MAX_TOKEN_NUM=2048

export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=last_lora

CHECKPOINT="checkpoints/qwen_1b_8b_large_d/v48-20260215-063905/checkpoint-1000"
OUTPUT="checkpoints/qwen_1b_mlang"

TRAIN_D="../ani/kv_150k_data/kv_batch_0_150000/train_mlang.json" # I used the lora for multilingual support
VAL_D="../ani/kv_150k_data/kv_batch_0_150000/val_mlang.json"

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model $CHECKPOINT \
    --tuner_type lora \
    --dataset $TRAIN_D \
    --val_dataset $VAL_D \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --merge_lora true \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 5000 \
    --output_dir $OUTPUT \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --report_to wandb
