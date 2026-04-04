from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.core.config import Config


class GeneralModelHandler(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        model_name: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = Config.DEVICE
        self.model_name = model_name
        self.ckpt_dir = Config.MODEL_CKPT_DIR / model_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_checkpoint(self, epoch, avg_loss):
        """Standardized saving so the GeneralTrainer can call it."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": avg_loss,
        }

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(ckpt, self.ckpt_dir / f"{self.model_name}_best.pt")
        torch.save(ckpt, self.ckpt_dir / f"{self.model_name}_epoch_{epoch}.pt")
    
    def load_checkpoint(self, checkpoint_path: Path | None = None, epoch: int | None = None):
        if checkpoint_path is None:
            checkpoint_path = self.ckpt_dir / f"{self.model_name}_best.pt"
        if epoch is not None:
            checkpoint_path = self.ckpt_dir / f"{self.model_name}_epoch_{epoch}.pt"

        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_loss = ckpt["loss"]
        return ckpt

    def train(self, dataloader, epochs):
        self.model.train()
        self.model.to(self.device)
        self.best_loss = float("inf")

        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", mininterval=1.0)
            for batch_idx, batch in enumerate(loop):
                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = self.train_step(batch)

                scaler.scale(loss).backward()

                if Config.GRAD_CLIP > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1
                
                lr = self.scheduler.get_last_lr()[0]
                
                if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                    loop.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss/num_batches:.4f}", lr=f"{lr:.2e}")
            
            self.scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)

            self.save_checkpoint(epoch, avg_loss)
    
