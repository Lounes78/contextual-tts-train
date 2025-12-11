import argparse
import os
import sys

# Set device visibility before importing torch to ensure strict isolation
if "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

import yaml
from pathlib import Path
from tqdm import tqdm
import optuna
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import wandb


from utils import (
    load_model, 
    load_tokenizers, 
    generate_audio, 
    WarmupDecayLR,
    validate,
    MIMI_SAMPLE_RATE,
)
from dataloaders import create_dataloaders

# DISABLE the buggy cuDNN SDPA backend -- need to check this later
torch.backends.cuda.enable_cudnn_sdp(False)

# ENABLE Flash Attention (best for GH200) and Mem Efficient
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def load_args():
    config_path = Path(__file__).parent / "train_config.yaml"

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    args = argparse.Namespace(**full_config)

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.gen_sentences.endswith(".txt") and os.path.exists(args.gen_sentences): # handle the case where the user provides a text file 
        args.gen_sentences = Path(args.gen_sentences)

    if args.train_from_scratch: # like this we are sure
        args.model_name_or_checkpoint_path = None
    else:
        # Auto-resume: Check if there are any checkpoints in the output directory
        # and resume from the latest one if found.
        checkpoints = list(args.output_dir.glob("model_*.pt"))
        latest_step = -1
        latest_ckpt = None
        
        for ckpt in checkpoints:
            # Skip special named checkpoints to prefer numbered steps for exact resume
            if ckpt.name in ["model_final.pt", "model_bestval.pt"]:
                continue
            try:
                # Expected format: model_{step}.pt
                step = int(ckpt.stem.split("_")[1])
                if step > latest_step:
                    latest_step = step
                    latest_ckpt = ckpt
            except (ValueError, IndexError):
                continue
        
        if latest_ckpt:
            print(f"Auto-resume: Found latest checkpoint in output_dir: {latest_ckpt}")
            args.model_name_or_checkpoint_path = str(latest_ckpt)

    return args


def train(args: argparse.Namespace, device: torch.device, rank: int, world_size: int, ddp_device_ids: list = None, trial: optuna.Trial = None):
    if rank == 0:
        assert wandb.run is not None, "WandB run must be initialized before calling train()"
    
    eff_batch_size = args.batch_size * args.grad_acc_steps * world_size

    model = load_model(
        model_name_or_checkpoint_path=args.model_name_or_checkpoint_path,
        device=device,
        decoder_loss_weight=args.decoder_loss_weight
    )
    
    if world_size > 1:
        model = DDP(model, device_ids=ddp_device_ids)

    trainloader, valloader = create_dataloaders(
        token_dataset_dir=args.data,
        batch_size=args.batch_size,
        infinite_train=False,
        load_in_memory=not args.partial_data_loading,
        num_workers=args.num_workers,
        num_replicas=world_size,
        rank=rank
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    total_steps = len(trainloader) * args.n_epochs
    scheduler = WarmupDecayLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        decay_type=args.lr_decay
    )

    step = 0
    if args.model_name_or_checkpoint_path and str(args.model_name_or_checkpoint_path).endswith(".pt"):
        print(f"Resuming training state from {args.model_name_or_checkpoint_path}")
        # Load to CPU first to avoid extra GPU memory usage
        checkpoint = torch.load(args.model_name_or_checkpoint_path, map_location="cpu")
        
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "step" in checkpoint:
            step = checkpoint["step"]
            print(f"Resuming from step {step}")

    train_losses = []
    
    state = {
        "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": 0,
        "best_val_loss": float("inf"),
    }

    if rank == 0:
        pbar = tqdm(total=total_steps, desc="Training")

    for epoch in range(args.n_epochs):
        for tokens, tokens_mask in trainloader:
            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)

            with autocast(device_type="cuda", enabled=args.use_amp, dtype=torch.bfloat16):
                loss = model(tokens, tokens_mask)
                loss = loss / args.grad_acc_steps
            
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() # update lr schedule by 1 step
            
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            # Clear cache to prevent OOM
            torch.cuda.empty_cache()


            if rank == 0 and args.log_every and step % args.log_every == 0:
                wandb.log(
                    {
                        "train_loss_avg": sum(train_losses) / len(train_losses),
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }, step=step
                    )
                train_losses = []
            # ------- SAVE ------- 
            if rank == 0 and args.save_every and (step % args.save_every == 0 or step == total_steps - 1):
                state["model"] = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(state, args.output_dir / f"model_{step}.pt")
                if step == total_steps - 1:
                    torch.save(state, args.output_dir / f"model_final.pt")
            
            # ------- VALIDATE -------
            if args.val_every and (step % args.val_every == 0 or step == total_steps - 1):
                if valloader is not None:
                    val_loss = validate(model, valloader, device, args.use_amp)
                    
                    # Aggregate validation loss across all ranks
                    if world_size > 1:
                        val_loss_tensor = torch.tensor(val_loss, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                        val_loss = val_loss_tensor.item()

                    if rank == 0:
                        wandb.log({"val_loss": val_loss}, step=step)

                        if val_loss < state["best_val_loss"]:
                            state["best_val_loss"] = val_loss
                            torch.save(state, args.output_dir / f"model_bestval.pt")
                            wandb.save(str(args.output_dir / f"wandb_bestval.pt"))

                        if trial is not None:
                            trial.report(val_loss, step)
                            if trial.should_prune():
                                wandb.finish()
                                pbar.close()
                                raise optuna.exceptions.TrialPruned()
            
                model.train()

                if rank == 0:
                    if valloader is not None:
                        pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
                    else:
                        pbar.set_postfix({"train_loss": f"{train_loss:.4f}"})

            else:
                if rank == 0:
                    pbar.set_postfix(
                        {"train_loss": f"{train_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                    )

            # # ------- GENERATE -------
            # if args.gen_every and step % args.gen_every == 0 and not (args.train_from_scratch and step == 0):
            #     gen_sentences = []
            #     if isinstance(args.gen_sentences, str):
            #         gen_sentences.append(args.gen_sentences)
            #     elif isinstance(args.gen_sentences, Path):
            #         with open(args.gen_sentences, "r") as f:
            #             gen_sentences = f.readlines()

            #     for i, sentence in enumerate(gen_sentences):
            #         audio = generate_audio(
            #             model,
            #             audio_tokenizer,
            #             text_tokenizer,
            #             sentence,
            #             args.gen_speaker,
            #             device,
            #             use_amp=args.use_amp
            #         )
                    
            #         wandb.log({f"audio_{i}": wandb.Audio(audio, sample_rate=MIMI_SAMPLE_RATE)}, step=step)
            #     model.train()

            if rank == 0:
                pbar.update(1)
            
            if total_steps and step >= total_steps:
                break
            step += 1
    
    if rank == 0:
        pbar.close()
    return state["best_val_loss"]



if __name__ == "__main__":
    args = load_args()

    # DDP setup
    ddp_device_ids = None
    if "LOCAL_RANK" in os.environ:
        # We already set CUDA_VISIBLE_DEVICES at the top of the file
        # so each process only sees one GPU, which is always cuda:0
        
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        
        device = torch.device("cuda:0")
        ddp_device_ids = [0]
        print(f"Rank {rank}/{world_size} initialized on {device} (Physical GPU {local_rank})")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")

    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"training_bs-{args.batch_size}x{args.grad_acc_steps}",
            notes=f"Config loaded from file",
            config=vars(args), # Namespace to dict for WandB
            reinit=args.wandb_reinit,
            dir=args.output_dir / "wandb",
        )

    final_val_loss = train(args, device, rank, world_size, ddp_device_ids)
    
    if rank == 0:
        wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()