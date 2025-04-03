from diffsynth import ModelManager, SD3ImagePipeline
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
import torch, os, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "True"

import lightning as pl
import wandb
import gc
import glob

class LightningModel(LightningModelForT2ILoRA):
    def __init__(
        self,
        torch_dtype=torch.float16, pretrained_weights=[], preset_lora_path=None,
        learning_rate=1e-4, use_gradient_checkpointing=True,
        lora_rank=4, lora_alpha=4, lora_target_modules="to_q,to_k,to_v,to_out", init_lora_weights="gaussian", pretrained_lora_path=None,
        quantize=None
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        # Load models
        model_manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        if quantize is None:
            model_manager.load_models(pretrained_weights)
        else:
            model_manager.load_models(pretrained_weights[1:])
            model_manager.load_model(pretrained_weights[0], torch_dtype=quantize)

        if preset_lora_path is not None:
            preset_lora_path = preset_lora_path.split(",")
            for path in preset_lora_path:
                model_manager.load_lora(path)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.init_lora_weights = init_lora_weights
        self.pretrained_lora_path = pretrained_lora_path
        
        self.pipe = SD3ImagePipeline.from_model_manager(model_manager)
        self.freeze_parameters()
        
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        for name, param in self.pipe.denoising_model().named_parameters():
            if "envmap" in name:
                print(name)
                with torch.no_grad():
                    param.data = param.to(torch.float32)
                param.requires_grad = True

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        
        # import bitsandbytes as bnb
        # optimizer = bnb.optim.Adam8bit(
        #     trainable_modules,
        #     lr=self.learning_rate,
        #     optim_bits=8,
        #     is_paged=True,
        # )
        return optimizer
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained models, separated by comma. For example, SD3: `models/stable_diffusion_3/sd3_medium_incl_clips_t5xxlfp16.safetensors`, SD3.5-large: `models/stable_diffusion_3/text_encoders/clip_g.safetensors,models/stable_diffusion_3/text_encoders/clip_l.safetensors,models/stable_diffusion_3/text_encoders/t5xxl_fp16.safetensors,models/stable_diffusion_3/sd3.5_large.safetensors`",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="a_to_qkv,b_to_qkv,norm_1_a.linear,norm_1_b.linear,a_to_out,b_to_out,ff_a.0,ff_a.2,ff_b.0,ff_b.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["float8_e4m3fn"],
        help="Whether to use quantization when training the model, and in which format.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Preset LoRA path.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of total timesteps. For turbo models, please set this parameter to the number of expected number of inference steps.",
    )
    parser = add_general_parsers(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = LightningModel(
        torch_dtype=torch.float32 if args.precision == "32" else torch.float16,
        pretrained_weights=args.pretrained_path.split(","),
        preset_lora_path=args.preset_lora_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
        lora_target_modules=args.lora_target_modules
    )
        
    # for name, param in model.pipe.denoising_model().named_parameters():
    #     if param.requires_grad and torch.isnan(param).any():
    #         print(f"NaN detected in parameter {name} after initializing LightningModel")
    #     if param.requires_grad and torch.isinf(param).any():
    #         print(f"Inf detected in parameter {name} after initializing LightningModel")
    
    resume_from_checkpoint = None
    launch_training_task(model, args, resume_from_checkpoint)