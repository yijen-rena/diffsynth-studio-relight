MODEL_PATH_ROOT=models/stable_diffusion_3.5_medium/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80
DATASET_ROOT=/ocean/projects/cis240058p/jzhang24/yehonathan_temp

START_TIME=$(date +%s)
START_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

python3 examples/train/stable_diffusion_3/train_sd3.py \
  --train_data_dir ${DATASET_ROOT}/material_anything_from_60000/transforms_train.json \
  --envmap_path ${DATASET_ROOT}/laval_haven \
  --num_train_epochs 20 \
  --resolution 256 \
  --train_batch_size 4 \
  --mixed_precision "fp16" \
  --learning_rate 1e-20 \
  --validation_prompts "an old-fashioned bow and arrow" \
  --use_8bit_adam
  # --use_wandb
  # --dataloader_num_workers 2
  # --dataset_path /ocean/projects/cis240058p/jzhang24/yehonathan_temp/material_anything_from_60000/train \
  # --dataset_path /ocean/projects/cis250002p/rju/datasets/metadata.csv \

END_TIME=$(date +%s)
END_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

DURATION_FORMATTED="${HOURS}h ${MINUTES}m ${SECONDS}s"

echo "Duration: $DURATION_FORMATTED"
