#!/usr/bin/env python3
import os,json
from datetime import datetime
from typing import Callable, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
from torch.amp import GradScaler, autocast
import deepspeed
from deepspeed.comm import get_rank

project_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ds_config = os.path.join(project_base,"ds_config.json")

class Trainer:
    """
    A modular PyTorch training engine.

    Args:
        model: a torch.nn.Module instance.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation/testing.
        optimizer: torch.optim.Optimizer instance.
        loss_fn: Callable that takes (outputs, targets) and returns a scalar loss.
        metric_fn: Callable that takes (outputs, targets) and returns a metric (e.g., accuracy).
        device: torch device string or torch.device.
        scheduler: Optional learning rate scheduler.
        early_stopping: Optional EarlyStopping instance.
        log_dir: Directory to write TensorBoard logs.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        device: str = 'cpu',
        scheduler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        stop_early: bool = True,
        log_dir: str = 'runs',
        label = None,
        mask_prob: Optional[float] = None,
        mask_len: Optional[int] = None,
        mask_test_prob: Optional[float] = None,
        mask_bin: Optional[int] = None,
        args: Optional[Any] = None,
    ):
        model = model.to(device)

        # initialize deepspeed engine
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            config_params=ds_config
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.stop_early = stop_early
        self.mask_prob = mask_prob
        self.mask_len = mask_len
        self.mask_bin = mask_bin
        self.mask_test_prob = mask_test_prob

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.label = label if label is not None else timestamp
        
        if get_rank() == 0:
            self.writer = SummaryWriter(os.path.join(log_dir, self.label))
        else:
            self.writer = None
        self.global_step = 0

        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.autocast = autocast
        else:
            self.autocast = nullcontext
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_metric = 0.0
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            self.model.zero_grad()
            
            with self.autocast('cuda'):
                inputs, _ = batch
                inputs = inputs.to(self.device)
                if self.mask_prob is not None and self.mask_len is not None:
                    targets = inputs.clone()
                    mask = torch.rand(inputs.size(0), self.mask_len, device=self.device) < self.mask_prob
                    mask = mask.unsqueeze(2).expand(-1, -1, self.mask_bin)
                    mask = mask.reshape(inputs.size(0), -1)
                    inputs[mask] = -100
                    outputs = self.model(inputs)
                    if self.mask_test_prob is not None:
                        if torch.rand(1).item() < self.mask_test_prob:
                            # Only calculate loss for masked positions:
                            outputs = outputs[mask]
                            targets = targets[mask]
                else:
                    outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            self.model.backward(loss)
            self.model.step()

            if self.writer is not None and self.metric_fn is not None:
                metric = self.metric_fn(outputs, targets)
                epoch_metric += metric
                self.writer.add_scalar('train/metric', metric, self.global_step)

            epoch_loss += loss.item()
            if self.writer is not None:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

            self.global_step += 1

        avg_loss = epoch_loss / num_batches
        if self.metric_fn is not None:
            avg_metric = epoch_metric / num_batches
            return {'loss': avg_loss, 'metric': avg_metric}
        else:
            return {'loss': avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        epoch_loss = 0.0
        epoch_metric = 0.0
        num_batches = len(self.val_loader)

        with self.autocast('cuda'):
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    if self.mask_prob is not None and self.mask_len is not None:
                        targets = inputs.clone()
                        mask = torch.rand(inputs.size(0), self.mask_len, device=self.device) < self.mask_prob
                        mask = mask.unsqueeze(2).expand(-1, -1, self.mask_bin)
                        mask = mask.reshape(inputs.size(0), -1)
                        inputs[mask] = -100
                        outputs = self.model(inputs)
                        if self.mask_test_prob is not None:
                            if torch.rand(1).item() < self.mask_test_prob:
                                # Only calculate loss for masked positions:
                                outputs = outputs[mask]
                                targets = targets[mask]
                    else:
                        outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    epoch_loss += loss.item()
                    if self.metric_fn is not None:
                        metric = self.metric_fn(outputs, targets)
                        epoch_metric += metric

        avg_loss = epoch_loss / num_batches
        avg_metric = epoch_metric / num_batches

        # logging
        if self.writer is not None:
            self.writer.add_scalar('val/loss', avg_loss, self.global_step)
            if self.metric_fn is not None:
                self.writer.add_scalar('val/metric', avg_metric, self.global_step)
                return {'loss': avg_loss, 'metric': avg_metric}
        return {'loss': avg_loss}

    def fit(
        self,
        num_epochs: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the training loop.

        Args:
            num_epochs: total number of epochs.
            checkpoint_dir: directory to save model checkpoints.
            checkpoint_interval: save checkpoint every N epochs.
            resume_from: path to checkpoint to resume from.

        Returns:
            best_stats: dictionary of best validation stats.
        """
        start_epoch = 1
        best_val_metric = float('-inf')
        best_stats = {}

        if resume_from is not None and os.path.isfile(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.scheduler and 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint.get('epoch', 1)
            best_val_metric = checkpoint.get('best_val_metric', best_val_metric)

        for epoch in range(start_epoch, num_epochs + 1):
            train_stats = self.train_epoch()
            val_stats = self.validate_epoch()

            if self.scheduler:
                self.scheduler.step()

            # early stopping
            if self.early_stopping:
                stop = self.early_stopping.test_early_stop(val_stats.get('loss', 0.0), self.model, val_stats.get('metric', 0.0))
                if self.stop_early and stop:
                    break

            # save best model
            val_metric = val_stats.get('metric', 0.0)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_stats = {'epoch': epoch, **val_stats}

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_path = os.path.join(checkpoint_dir, f'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_metric': best_val_metric,
                    }, save_path)

            # periodic checkpoint
            if checkpoint_dir and epoch % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                cp_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                }, cp_path)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_stats['loss']:.4f}, "
                  f"Train Metric: {train_stats.get('metric', 0.0):.4f}, "
                  f"Val Loss: {val_stats.get('loss', 0.0):.4f}, "
                  f"Val Metric: {val_stats.get('metric', 0.0):.4f}")

        best_stats['last_metric'] = val_stats.get('metric', 0.0)
        best_stats['last_loss'] = val_stats.get('loss', 0.0)
        best_stats['avg_acc'] = self.early_stopping.average_acc if self.early_stopping else None
        best_stats['best_acc'] = self.early_stopping.best_acc if self.early_stopping else None
        best_stats['best_epoch'] = self.early_stopping.best_epoch if self.early_stopping else None

        return best_stats