python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path data/DL3DV-10K-Samples \
  --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 25 \
  --height 480 \
  --width 832