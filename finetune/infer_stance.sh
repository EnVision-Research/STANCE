#!/usr/bin/env bash
echo "Starting Validation script..."
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=YES

NAME="video_generation_runs"
ITERS=165000

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"  # ["sft", "lora"]
    --use_forcing false
    --average_ins_flow true
    --use_mass true
    --use_depth true
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir /YOUR/OWN/PRETRAINED/PATH/$NAME/$ITERS
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "not_important"
    --caption_column "not_important"
    --video_column "YOUR_FIND_VIDEO_PATH.txt"
    # --video_column "videos_1obj_val.txt"
    --train_resolution "49x256x256"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 50 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000 # save checkpoint every x steps
    --checkpointing_limit 15 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --load_pretrained_weight "/YOUR/OWN/PRETRAINED/PATH/$NAME/checkpoint-$ITERS/pytorch_model/mp_rank_00_model_states.pt" 
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "NotImportant"
    --validation_steps 1000  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
    --num_validation_samples 1
    --is_validation true
)

accelerate launch --num_processes=1 train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
