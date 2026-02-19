#!/bin/bash

BASE="/home/ubuntu/distil/checkpoints/qwen_1b_8b_large_d/v48-20260215-063905/checkpoint-1000"
ADAPTER_1="/home/ubuntu/distil/checkpoints/qwen_1b_mlang/v3-20260216-183608/checkpoint-5250"
FINAL="/home/ubuntu/distil/checkpoints/qwen_1b_full"

echo "Merging Adapter into Base..."
swift export \
    --model "$BASE" \
    --adapters "$ADAPTER_1" \
    --merge_lora true \
    --output_dir "$FINAL"

echo "Final merged model is in: $FINAL"
