#!/usr/bin/env bash
echo "Starting training script..."
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DISABLE_TELEMETRY=YES

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox1.5-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "lora"  # ["sft", "lora"] debug with lora, training with sft require multi-GPUs
)

# Output Configuration
OUTPUT_ARGS=(
    # v1: joint_SegRgb_dense_0.5NC_3dflow_3frames_vitb
    --output_dir "/data/user/zmai090/C/video_generation_runs_realistic_demo_demo_demo_ft"
    --report_to "tensorboard"  # ["tensorboard", "wandb"]
)

# Data Configuration
DATA_ARGS=(
    --data_root "not important"
    --caption_column "not important"
    --train_resolution "49x256x256"  # (frames x height x width), frames should be 8N+1
    # --video_column "videos_2-5objs_train.txt"
    --video_column "demo_demo_demo.txt"
)

# Training Configuration
TRAIN_ARGS=(
    --learning_rate 1e-6
    --train_epochs 250000   # 1obj: 20 5w, 2-5 objs: 61 15w
    --seed 45 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"]
    --use_mass true
    --use_depth true
    --average_ins_flow true
    # --use_ema true
    # --ema_decay 0.9999
    # --save_ema_as_main false  # true: save EMA as main weights, false: save training weights as main
    # --max_grad_norm 1.0
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 200 # save checkpoint every x steps
    --checkpointing_limit 20 # maximum number of checkpoints to keep, after which the oldest one is deleted
    #--resume_from_checkpoint "PATH/TO/YOUR/CHECKPOINT"
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "NotImportant"
    --validation_steps 200  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
    --num_validation_samples 4
)

# Combine all arguments and launch training (lora)
# accelerate launch --num_processes=1 train.py \
#     "${MODEL_ARGS[@]}" \
#     "${OUTPUT_ARGS[@]}" \
#     "${DATA_ARGS[@]}" \
#     "${TRAIN_ARGS[@]}" \
#     "${SYSTEM_ARGS[@]}" \
#     "${CHECKPOINT_ARGS[@]}" \
#     "${VALIDATION_ARGS[@]}"

# sft
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"

# --config_file 0.yaml
# bash train_motion.sh
# bash train_motion.sh > log.txt 2>&1 &