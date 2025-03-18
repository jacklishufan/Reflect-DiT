#! /bin/bash

OUTPUT_DIR="training_outputs/reflect-dit"
MODEL_NAME=ckpts/Sana_1600M_1024px_MultiLing_diffusers
INSTANCE_DIR=data/dit/object_self_correct_cleaned.csv
accelerate launch --num_processes 8 --main_process_port 29500 --gpu_ids 0,1,2,3,4,5,6,7 \
  --config_file train_scripts/accelerate_config.json \
  train_scripts/train_sana_self_correction.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=500 \
  --max_train_steps=5000 \
  --val_every 500 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 5 \
  --validation_prompt="a photo of a cell phone right of a chair" \
  --seed="0" \
  --no_lora \
  --method reflection \
  --max_negatives 3 \
  ${@} \
  --weighting_scheme logit_normal \
  --optimizer came \