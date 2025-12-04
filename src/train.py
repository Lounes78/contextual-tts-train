import argparse
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import optuna
import torch
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

# DISABLE the buggy cuDNN SDPA backend
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

    return args


def train(args: argparse.Namespace, device: torch.device, trial: optuna.Trial = None):
    assert wandb.run is not None, "WandB run must be initialized before calling train()"
    
    eff_batch_size = args.batch_size * args.grad_acc_steps

    model = load_model(
        model_name_or_checkpoint_path=args.model_name_or_checkpoint_path,
        device=device,
        decoder_loss_weight=args.decoder_loss_weight
    )
    text_tokenizer, audio_tokenizer = load_tokenizers(device)

    print(f"Initializing dataloaders from {args.data}")
    trainloader, valloader = create_dataloaders(
        token_dataset_dir=args.data, 
        batch_size=args.batch_size, 
        infinite_train=False,
        load_in_memory=not args.partial_data_loading,
        num_workers=args.num_workers
    )

    if trainloader is None:
        raise ValueError("Training dataloader is None.")

    total_steps = args.n_epochs * len(trainloader) # One update per batch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = WarmupDecayLR(optimizer, args.warmup_steps, total_steps, args.lr_decay)
    scaler = GradScaler(enabled=args.use_amp)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "effective_batch_size": eff_batch_size,
        "args": vars(args), 
        "best_val_loss": float("inf"),
    }   

    # training loop
    step = 0
    train_losses = []
    pbar = tqdm(total=total_steps, desc="Training" if trial is None else f"Trial {trial.number}")
    model.train()

    for epoch in range(args.n_epochs):
        for tokens, tokens_mask in trainloader:
            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)

            with autocast(device_type=str(device), enabled=args.use_amp, dtype=torch.bfloat16):
                loss = model(tokens, tokens_mask)
                loss = loss / args.grad_acc_steps
            
            scaler.scale(loss).backward() # backprop safely in mixed precision by scaling the loss first

            if (step + 1) % args.grad_acc_steps == 0:
                scaler.unscale_(optimizer) # unscale the gradients before clipping
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer) # aka optimizer.step() safely 
                scaler.update() # adjust the scale for next iteration
                optimizer.zero_grad()
                scheduler.step() # update lr schedule by 1 step
            
            train_loss = loss.item()
            train_losses.append(train_loss)


            if args.log_every and step % args.log_every == 0:
                wandb.log(
                    {
                        "train_loss_avg": sum(train_losses) / len(train_losses),
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }, step=step
                    )
                train_losses = []
            # ------- SAVE ------- 
            if args.save_every and (step % args.save_every == 0 or step == total_steps - 1):
                state["model"] = model.state_dict()
                torch.save(state, args.output_dir / f"model_{step}.pt")
                if step == total_steps - 1:
                    torch.save(state, args.output_dir / f"model_final.pt")
            
            # ------- VALIDATE -------
            if args.val_every and (step % args.val_every == 0 or step == total_steps - 1):
                if valloader is not None:
                    val_loss = validate(model, valloader, device, args.use_amp)
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

                if valloader is not None:
                    pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
                else:
                    pbar.set_postfix({"train_loss": f"{train_loss:.4f}"})

            else:
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

            pbar.update(1)
            if total_steps and step >= total_steps:
                break
            step += 1
    
    pbar.close()
    return state["best_val_loss"]



if __name__ == "__main__":
    args = load_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"training_bs-{args.batch_size}x{args.grad_acc_steps}",
        notes=f"Config loaded from file",
        config=vars(args), # Namespace to dict for WandB
        reinit=args.wandb_reinit,
        dir=args.output_dir / "wandb",
    )

    final_val_loss = train(args, device)
    wandb.finish()