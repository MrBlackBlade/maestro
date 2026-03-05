from abc import ABC, abstractmethod

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
            "loss": avg_loss,
        }

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(ckpt, self.ckpt_dir / f"{self.model_name}_best.pt")
        torch.save(ckpt, self.ckpt_dir / f"{self.model_name}_epoch_{epoch}.pt")
    
    def train(self, dataloader, epochs):
        self.model.train()
        self.model.to(self.device)
        self.best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", mininterval=1.0)
            for batch_idx, batch in enumerate(loop):
                loss = self.train_step(batch)
                self.optimizer.zero_grad()

                loss.backward()
                epoch_loss += loss.item()
                num_batches += 1
                self.optimizer.step()
                self.scheduler.step()
                
                lr = self.scheduler.get_last_lr()[0]
                
                if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                    loop.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss/num_batches:.4f}", lr=f"{lr:.2e}")
            
            avg_loss = epoch_loss / max(num_batches, 1)

            self.save_checkpoint(epoch, avg_loss)
    
