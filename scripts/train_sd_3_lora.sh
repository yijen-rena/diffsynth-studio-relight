MODEL_PATH_ROOT=models/stable_diffusion_3/models--stabilityai--stable-diffusion-3-medium/snapshots/19b7f516efea082d257947e057e6f419e26fd497

python examples/train/stable_diffusion_3/train_sd3_lora.py \
  --pretrained_path ${MODEL_PATH_ROOT}/sd3_medium_incl_clips.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "16" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing