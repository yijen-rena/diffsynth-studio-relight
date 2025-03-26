MODEL_PATH_ROOT=models/stable_diffusion_xl/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b

python examples/train/stable_diffusion_xl/train_sdxl_lora.py \
  --pretrained_path ${MODEL_PATH_ROOT}/sd_xl_base_1.0.safetensors \
  --dataset_path data/dog \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "32" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing