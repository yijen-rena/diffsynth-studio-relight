MODEL_PATH_ROOT=models/stable_diffusion_3.5_large_turbo/models--stabilityai--stable-diffusion-3.5-large-turbo/snapshots/ec07796fc06b096cc56de9762974a28f4c632eda

python examples/train/stable_diffusion_3/train_sd3_lora.py \
  --pretrained_path ${MODEL_PATH_ROOT}/text_encoders/clip_g.safetensors,${MODEL_PATH_ROOT}/text_encoders/clip_l.safetensors,${MODEL_PATH_ROOT}/text_encoders/t5xxl_fp16.safetensors,${MODEL_PATH_ROOT}/sd3.5_large.safetensors \
  --dataset_path data/dalle3_weird_animals \
  --output_path ./models \
  --max_epochs 1 \
  --steps_per_epoch 500 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 4 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4