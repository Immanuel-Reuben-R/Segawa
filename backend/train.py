import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CornellMovieDataset
from model import SegawaModel, device

# ---------------- Hyperparameters ----------------
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 3e-4      # 1e-3 is too hot for a transformer
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05        # first 5% of steps ramp LR up from ~0
MIN_LR_FRAC = 0.10        # cosine decays to 10% of peak LR
GRAD_CLIP = 1.0
PATIENCE = 4              # stop after N epochs with no val-loss improvement
MAX_LENGTH = 60
VOCAB_SIZE = 8000
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    # No weight decay on biases / LayerNorm / 1-D params
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)


def build_scheduler(optimizer, total_steps):
    warmup = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, pad_idx, use_amp, dev_type):
    """Token-weighted loss/accuracy so every real token counts equally."""
    model.eval()
    sum_loss = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss, total_tokens, correct = 0.0, 0, 0

    for inputs, targets in tqdm(loader, leave=False, desc="Val"):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
        outputs = outputs.float().reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)

        mask = targets != pad_idx
        total_loss += sum_loss(outputs, targets).item()
        total_tokens += mask.sum().item()
        correct += ((outputs.argmax(dim=1) == targets) & mask).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    acc = 100.0 * correct / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, acc, ppl


def train():
    print("Initializing Segawa Training Pipeline...")
    set_seed(SEED)

    # 1. Data
    dataset = CornellMovieDataset(
        r"datasets/movie_lines.txt",
        r"datasets/movie_conversations.txt",
        r"datasets/input.txt",
        max_length=MAX_LENGTH,
        vocab_size=VOCAB_SIZE,
    )

    train_idx, val_idx = dataset.split_by_conversation(val_frac=0.05, seed=SEED)
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
    train_size, val_size = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 2. Model
    print(f"Loading Model to {device}...")
    model = SegawaModel(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # 3. Optimizer / scheduler / loss
    dev_type = torch.device(device).type
    use_amp = dev_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = build_optimizer(model)
    total_steps = EPOCHS * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)

    os.makedirs("checkpoints", exist_ok=True)
    best_val, bad_epochs = float("inf"), 0

    print(f"Beginning Training Loop! (Train: {train_size}, Val: {val_size})")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        loop = tqdm(train_loader, leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
            outputs = outputs.float().reshape(-1, outputs.shape[-1])
            loss = criterion(outputs, targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}] Train")
            loop.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train = total_train_loss / len(train_loader)
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, dataset.PAD_IDX, use_amp, dev_type)

        print(f"Epoch {epoch + 1} finished!")
        print(f"  Train Loss: {avg_train:.4f} | Train PPL: {math.exp(avg_train):.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val PPL: {val_ppl:.2f}")

        # Save best + latest
        torch.save(model.state_dict(), "checkpoints/segawa_last.pth")
        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), "checkpoints/segawa_best.pth")
            print("  New best model saved (checkpoints/segawa_best.pth)\n")
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{PATIENCE})\n")
            if bad_epochs >= PATIENCE:
                print("Early stopping: validation loss stopped improving.")
                break

        if val_ppl < 50.0:
            print("Target perplexity < 50 achieved!")

    print(f"Done. Best val loss: {best_val:.4f} (PPL {math.exp(best_val):.2f})")


if __name__ == "__main__":
    train()