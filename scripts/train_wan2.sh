python examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture full \
  --dataset_path data/dalle3_weird_animals \
  --output_path ./models \
  --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 \
  # --strict