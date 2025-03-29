MODEL_PATH_ROOT=models/stable_diffusion_3.5_medium/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80

START_TIME=$(date +%s)
START_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

python examples/train/stable_diffusion_3/train_sd3_lora.py \
  --pretrained_path ${MODEL_PATH_ROOT}/text_encoders/clip_g.safetensors,${MODEL_PATH_ROOT}/text_encoders/clip_l.safetensors,${MODEL_PATH_ROOT}/text_encoders/clip_g.safetensors,${MODEL_PATH_ROOT}/text_encoders/t5xxl_fp16.safetensors,${MODEL_PATH_ROOT}/sd3.5_medium.safetensors \
  --dataset_path /ocean/projects/cis240058p/jzhang24/yehonathan_temp/material_anything_from_60000/train \
  --output_path ./models \
  --max_epochs 5 \
  --steps_per_epoch 100 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "16-mixed" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing \
  --use_wandb
  # --dataloader_num_workers 4 \

END_TIME=$(date +%s)
END_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

DURATION_FORMATTED="${HOURS}h ${MINUTES}m ${SECONDS}s"

echo "Duration: $DURATION_FORMATTED"
