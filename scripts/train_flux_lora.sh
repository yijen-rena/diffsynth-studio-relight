MODEL_PATH_ROOT=models/FLUX/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44

python examples/train/flux/train_flux_lora.py \
  --pretrained_text_encoder_path ${MODEL_PATH_ROOT}/text_encoder/model.safetensors \
  --pretrained_text_encoder_2_path ${MODEL_PATH_ROOT}/text_encoder_2 \
  --pretrained_dit_path ${MODEL_PATH_ROOT}/flux1-dev.safetensors \
  --pretrained_vae_path ${MODEL_PATH_ROOT}/ae.safetensors \
  --dataset_path data/dalle3_weird_animals \
  --output_path ./models/FLUX/FLUX_lora \
  --max_epochs 1 \
  --steps_per_epoch 100 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "bf16" \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --use_gradient_checkpointing \
  --align_to_opensource_format \
  --quantize "float8_e4m3fn"