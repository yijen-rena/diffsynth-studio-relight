import lightning as pl
from peft import LoraConfig, inject_adapter_in_model
import torch, os
from ..data.simple_text_image import TextImageDataset
from ..data.data_utils import DatasetTextAndEnvmapToImage, read_hdr
from modelscope.hub.api import HubApi
from ..models.utils import load_state_dict

import wandb

class LightningModelForT2ILoRA(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        state_dict_converter=None,
    ):
        super().__init__()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = state_dict_converter
        self.lora_alpha = None


    def load_models(self):
        # This function is implemented in other modules
        self.pipe = None


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    
    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="to_q,to_k,to_v,to_out",
        init_lora_weights="gaussian",
        pretrained_lora_path=None,
        state_dict_converter=None,
        lora_weight_precision=torch.float32
    ):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)

        # for param in model.parameters():
        for name, param in model.named_parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(lora_weight_precision)

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")


    # def training_step(self, batch, batch_idx):
    #     # Data
    #     text, image = batch["text"], batch["image"]

    #     # Prepare input parameters
    #     self.pipe.device = self.device
    #     prompt_emb = self.pipe.encode_prompt(text, positive=True)
    #     if "latents" in batch:
    #         latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
    #     else:
    #         latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
    #     noise = torch.randn_like(latents)
    #     timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
    #     timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
    #     extra_input = self.pipe.prepare_extra_input(latents)
    #     noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
    #     training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

    #     # Compute loss
    #     noise_pred = self.pipe.denoising_model()(
    #         noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
    #         use_gradient_checkpointing=self.use_gradient_checkpointing
    #     )
    #     loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    #     loss = loss * self.pipe.scheduler.training_weight(timestep)

    #     # Record log
    #     self.log("train_loss", loss, prog_bar=True)
    #     # self.log("grad_norm", torch.norm(torch.stack([torch.norm(p.grad) for p in self.pipe.denoising_model().parameters() if p.grad is not None]), dim=0), prog_bar=True)
    #     return loss
    
    def training_step(self, batch, batch_idx):
        text, image, envmap, dir_embeds, T = batch["text"], batch["image"], batch["envmap"], batch["dir_embeds"], batch["T"]

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True) # [1, 154, 4096], [1, 2048]
        envmap_emb = self.pipe.encode_envir_map(envmap.to(dtype=self.pipe.torch_dtype, device=self.device))
        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **envmap_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        # self.log("grad_norm", torch.norm(torch.stack([torch.norm(p.grad) for p in self.pipe.denoising_model().parameters() if p.grad is not None]), dim=0), prog_bar=True)
        return loss


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
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        if self.state_dict_converter is not None:
            lora_state_dict = self.state_dict_converter(lora_state_dict, alpha=self.lora_alpha)
        checkpoint.update(lora_state_dict)


class ValidationImageCallback(pl.Callback):
    def __init__(self, validation_prompts, cfg_scale=7.5):
        super().__init__()
        self.validation_prompts = validation_prompts
        self.cfg_scale = cfg_scale
    
    def on_train_epoch_end(self, trainer, pl_module):
        logger = trainer.logger
        
        with torch.no_grad():
            for prompt in self.validation_prompts:
                image = pl_module.pipe(
                    prompt=prompt,
                    cfg_scale=self.cfg_scale,
                )
                
                caption = f"Epoch {trainer.current_epoch}: {prompt}"
                logger.experiment.log({
                    f"validation_images/{prompt}": [wandb.Image(image, caption=caption)]
                })
                
        pl_module.pipe.scheduler.set_timesteps(1000, training=True)

def add_general_parsers(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset config.",
    )
    parser.add_argument(
        "--envmap_path",
        type=str,
        default=None,
        required=True,
        help="The path to environment maps.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="Whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16", "16-mixed", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--modelscope_model_id",
        type=str,
        default=None,
        help="Model ID on ModelScope (https://www.modelscope.cn/). The model will be uploaded to ModelScope automatically if you provide a Model ID.",
    )
    parser.add_argument(
        "--modelscope_access_token",
        type=str,
        default=None,
        help="Access key on ModelScope (https://www.modelscope.cn/). Required if you want to upload the model to ModelScope.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Whether to use WandB logger.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=["an old-fashioned bow"],
        help="Validation prompts. Required if use_wandb is True.",
    )
    
    return parser


def launch_training_task(model, args, resume_from_checkpoint=None):
    # dataset and data loader
    # dataset = TextImageDataset(
    #     args.dataset_path,
    #     steps_per_epoch=args.steps_per_epoch * args.batch_size,
    #     height=args.height,
    #     width=args.width,
    #     center_crop=args.center_crop,
    #     random_flip=args.random_flip
    # )
    dataset = DatasetTextAndEnvmapToImage(
        args.dataset_config_path,
        steps_per_epoch=args.steps_per_epoch,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        envmap_path=args.envmap_path
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers
    )
    # train
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="diffsynth_studio", 
            name="diffsynth_studio",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    elif args.use_wandb:
        import datetime
        from pytorch_lightning.loggers import WandbLogger
        
        os.environ["WANDB_DIR"] = args.output_path
        wandb_logger = WandbLogger(
            project="diffsynth_studio-relight",
            name=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
            save_dir=args.output_path,
        )
        logger = [wandb_logger]
    else:
        logger = None
        
    callbacks = [
        pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1),
    ]
    
    if args.use_wandb:
        validation_callback = ValidationImageCallback(
            args.validation_prompts,
            cfg_scale=7.5
        )
        callbacks.append(validation_callback)
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
    )
    if resume_from_checkpoint is not None:
        trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=resume_from_checkpoint)
    else:
        # import pdb; pdb.set_trace()
        trainer.fit(model=model, train_dataloaders=train_loader) # FIXME: make sure len(train_loader) is 519 is correct?

    # Upload models
    if args.modelscope_model_id is not None and args.modelscope_access_token is not None:
        print(f"Uploading models to modelscope. model_id: {args.modelscope_model_id} local_path: {trainer.log_dir}")
        with open(os.path.join(trainer.log_dir, "configuration.json"), "w", encoding="utf-8") as f:
            f.write('{"framework":"Pytorch","task":"text-to-image-synthesis"}\n')
        api = HubApi()
        api.login(args.modelscope_access_token)
        api.push_model(model_id=args.modelscope_model_id, model_dir=trainer.log_dir)
