START_TIME=$(date +%s)
START_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

python examples/train/stable_diffusion/train_sd2.1.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --train_data_dir data/dalle3_weird_animals \
  --output_dir ./models \
  --resolution 256 \
  --mixed_precision fp16 \
  --max_train_steps 100 \
  --num_train_epochs 1 \
  --train_batch_size 48 \
  --learning_rate 1e-20 \
  --validation_prompts "a turqouise elephant" \
  # --use_8bit_adam \
  # --pretrained_model_name_or_path ${MODEL_PATH_ROOT} \

END_TIME=$(date +%s)
END_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

DURATION_FORMATTED="${HOURS}h ${MINUTES}m ${SECONDS}s"

echo "Duration: $DURATION_FORMATTED"