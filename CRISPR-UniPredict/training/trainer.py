"""
Main Trainer Class for CRISPR-UniPredict
Handles training, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np

from models.crispr_unipredict import CRISPRUniPredict
from utils.losses import MultiTaskLoss
from utils.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main Trainer class for CRISPR-UniPredict
    
    Handles:
    - Training and validation loops
    - Checkpoint saving/loading
    - Logging (TensorBoard, WandB)
    - Mixed precision training
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling
    
    Example:
        >>> from configs.config_loader import ConfigLoader
        >>> from utils.preprocessing.dataloader_factory import create_dataloaders
        >>> from models.crispr_unipredict import CRISPRUniPredict
        >>> 
        >>> config_loader = ConfigLoader('configs/model_config.yaml')
        >>> config = config_loader.config
        >>> dataloaders = create_dataloaders(config)
        >>> 
        >>> model = CRISPRUniPredict(device='cuda')
        >>> trainer = Trainer(model, dataloaders, config)
        >>> history = trainer.train()
    """
    
    def __init__(self,
                 model: CRISPRUniPredict,
                 dataloaders: Dict,
                 config,
                 resume_from: Optional[str] = None):
        """
        Initialize Trainer
        
        Args:
            model: CRISPRUniPredict model instance
            dataloaders: Dictionary with 'train', 'val', 'test' dataloaders
            config: Configuration object
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.device = 'cuda' if config.device.use_cuda else 'cpu'
        
        # Setup directories
        self.checkpoint_dir = Path(config.logging.checkpoint_dir)
        self.log_dir = Path(config.logging.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_optimizers()
        self._init_loss_function()
        self._init_scheduler()
        self._init_logging()
        self._init_metrics()
        
        # Mixed precision training
        self.use_amp = config.device.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
        self.training_history = {
            'train': [],
            'val': []
        }
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info("Trainer initialized successfully")
    
    def _init_optimizers(self) -> None:
        """Initialize optimizers for encoder and heads"""
        # Separate parameters
        encoder_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'head' in name or 'fusion' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)
        
        # Create optimizers
        if self.config.training.optimizer == 'AdamW':
            self.optimizer_encoder = optim.AdamW(
                encoder_params,
                lr=self.config.training.learning_rate_encoder,
                weight_decay=self.config.training.weight_decay
            )
            self.optimizer_heads = optim.AdamW(
                head_params,
                lr=self.config.training.learning_rate_heads,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        logger.info(
            f"Optimizers initialized:\n"
            f"  Encoder: {len(encoder_params)} parameters\n"
            f"  Heads: {len(head_params)} parameters"
        )
    
    def _init_loss_function(self) -> None:
        """Initialize multi-task loss function"""
        self.criterion = MultiTaskLoss(
            on_target_weight=self.config.training.loss.loss_weights['on_target'],
            off_target_weight=self.config.training.loss.loss_weights['off_target'],
            learnable_weights=False,
            on_target_loss_fn=self.config.training.loss.on_target_loss,
            off_target_loss_fn=self.config.training.loss.off_target_loss
        ).to(self.device)
        
        logger.info("Loss function initialized")
    
    def _init_scheduler(self) -> None:
        """Initialize learning rate scheduler"""
        scheduler_type = self.config.training.scheduler.type
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_heads,
                mode='min',
                factor=self.config.training.scheduler.factor,
                patience=self.config.training.scheduler.patience,
                min_lr=self.config.training.scheduler.min_lr
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_heads,
                T_max=self.config.training.epochs
            )
        else:
            self.scheduler = None
        
        logger.info(f"Scheduler initialized: {scheduler_type}")
    
    def _init_logging(self) -> None:
        """Initialize logging (TensorBoard, WandB)"""
        try:
            import numpy as np
            # NumPy 2.x removed np.bool8; older tensorboard/protobuf expect it
            if not hasattr(np, "bool8"):
                np.bool8 = np.bool_

            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard writer initialized at {self.log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.writer = None
        
        # WandB initialization (optional)
        if self.config.logging.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    config=self._config_to_dict(),
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logger.info("WandB initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
    
    def _init_metrics(self) -> None:
        """Initialize metrics calculator"""
        self.metrics_calculator = MetricsCalculator()
    
    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        from dataclasses import asdict
        try:
            return asdict(self.config)
        except:
            return {}
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'on_target_loss': 0.0,
            'off_target_loss': 0.0,
            'batch_count': 0
        }
        
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}",
            disable=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            sgrna_onehot = batch['sgrna_onehot'].to(self.device)
            sgrna_label = batch['sgrna_label'].to(self.device)
            on_target_score = batch['on_target_score'].to(self.device)
            off_target_label = batch['off_target_label'].to(self.device)
            on_target_mask = batch['on_target_mask'].to(self.device)
            off_target_mask = batch['off_target_mask'].to(self.device)
            sgrna_strs = batch.get('sgrna_strs')
            target_strs = batch.get('target_strs')

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    on_target_pred, off_target_pred = self.model(
                        sgrna_onehot, sgrna_label, task_type='both',
                        sgrna_strs=sgrna_strs, target_strs=target_strs
                    )
                    
                    loss, loss_dict = self.criterion(
                        on_target_pred=on_target_pred,
                        off_target_pred=off_target_pred,
                        on_target_target=on_target_score,
                        off_target_target=off_target_label,
                        on_target_mask=on_target_mask,
                        off_target_mask=off_target_mask
                    )
            else:
                on_target_pred, off_target_pred = self.model(
                    sgrna_onehot, sgrna_label, task_type='both',
                    sgrna_strs=sgrna_strs, target_strs=target_strs
                )
                
                loss, loss_dict = self.criterion(
                    on_target_pred=on_target_pred,
                    off_target_pred=off_target_pred,
                    on_target_target=on_target_score,
                    off_target_target=off_target_label,
                    on_target_mask=on_target_mask,
                    off_target_mask=off_target_mask
                )
            
            # Backward pass
            self.optimizer_encoder.zero_grad()
            self.optimizer_heads.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer_encoder)
                self.scaler.unscale_(self.optimizer_heads)
            else:
                loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip
            )
            
            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer_encoder)
                self.scaler.step(self.optimizer_heads)
                self.scaler.update()
            else:
                self.optimizer_encoder.step()
                self.optimizer_heads.step()
            
            # Update metrics
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['on_target_loss'] += loss_dict['on_target_loss'].item()
            epoch_metrics['off_target_loss'] += loss_dict['off_target_loss'].item()
            epoch_metrics['batch_count'] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'on_target': loss_dict['on_target_loss'].item(),
                'off_target': loss_dict['off_target_loss'].item()
            })
            
            # Log batch metrics
            if (batch_idx + 1) % self.config.logging.print_frequency == 0:
                global_step = self.current_epoch * len(self.dataloaders['train']) + batch_idx
                
                if self.writer:
                    self.writer.add_scalar('train/loss', loss.item(), global_step)
                    self.writer.add_scalar('train/on_target_loss', loss_dict['on_target_loss'].item(), global_step)
                    self.writer.add_scalar('train/off_target_loss', loss_dict['off_target_loss'].item(), global_step)
        
        # Average metrics
        for key in ['total_loss', 'on_target_loss', 'off_target_loss']:
            epoch_metrics[key] /= max(epoch_metrics['batch_count'], 1)
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        val_metrics = {
            'total_loss': 0.0,
            'on_target_loss': 0.0,
            'off_target_loss': 0.0,
            'batch_count': 0
        }
        
        # Collect predictions for detailed metrics
        all_on_target_pred = []
        all_on_target_target = []
        all_off_target_pred = []
        all_off_target_target = []
        
        with torch.no_grad():
            pbar = tqdm(
                self.dataloaders['val'],
                desc="Validating",
                disable=False
            )
            
            for batch in pbar:
                # Move to device
                sgrna_onehot = batch['sgrna_onehot'].to(self.device)
                sgrna_label = batch['sgrna_label'].to(self.device)
                on_target_score = batch['on_target_score'].to(self.device)
                off_target_label = batch['off_target_label'].to(self.device)
                on_target_mask = batch['on_target_mask'].to(self.device)
                off_target_mask = batch['off_target_mask'].to(self.device)
                sgrna_strs = batch.get('sgrna_strs')
                target_strs = batch.get('target_strs')

                # Forward pass
                if self.use_amp:
                    with autocast():
                        on_target_pred, off_target_pred = self.model(
                            sgrna_onehot, sgrna_label, task_type='both',
                            sgrna_strs=sgrna_strs, target_strs=target_strs
                        )
                        
                        loss, loss_dict = self.criterion(
                            on_target_pred=on_target_pred,
                            off_target_pred=off_target_pred,
                            on_target_target=on_target_score,
                            off_target_target=off_target_label,
                            on_target_mask=on_target_mask,
                            off_target_mask=off_target_mask
                        )
                else:
                    on_target_pred, off_target_pred = self.model(
                        sgrna_onehot, sgrna_label, task_type='both',
                        sgrna_strs=sgrna_strs, target_strs=target_strs
                    )
                    
                    loss, loss_dict = self.criterion(
                        on_target_pred=on_target_pred,
                        off_target_pred=off_target_pred,
                        on_target_target=on_target_score,
                        off_target_target=off_target_label,
                        on_target_mask=on_target_mask,
                        off_target_mask=off_target_mask
                    )
                
                # Update metrics
                val_metrics['total_loss'] += loss.item()
                val_metrics['on_target_loss'] += loss_dict['on_target_loss'].item()
                val_metrics['off_target_loss'] += loss_dict['off_target_loss'].item()
                val_metrics['batch_count'] += 1
                
                # Collect predictions
                all_on_target_pred.append(on_target_pred[on_target_mask].cpu())
                all_on_target_target.append(on_target_score[on_target_mask].cpu())
                all_off_target_pred.append(off_target_pred[off_target_mask].cpu())
                all_off_target_target.append(off_target_label[off_target_mask].cpu())
        
        # Average losses
        for key in ['total_loss', 'on_target_loss', 'off_target_loss']:
            val_metrics[key] /= max(val_metrics['batch_count'], 1)
        
        # Compute detailed metrics
        if all_on_target_pred:
            on_target_pred_all = torch.cat(all_on_target_pred)
            on_target_target_all = torch.cat(all_on_target_target)
            on_target_detailed = self.metrics_calculator.compute_on_target_metrics(
                on_target_pred_all, on_target_target_all
            )
            val_metrics.update({f'on_target_{k}': v for k, v in on_target_detailed.items()})
        
        if all_off_target_pred:
            off_target_pred_all = torch.cat(all_off_target_pred)
            off_target_target_all = torch.cat(all_off_target_target)
            off_target_detailed = self.metrics_calculator.compute_off_target_metrics(
                off_target_pred_all, off_target_target_all
            )
            val_metrics.update({f'off_target_{k}': v for k, v in off_target_detailed.items()})
        
        return val_metrics
    
    def train(self) -> Dict:
        """
        Main training loop
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.training_history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.training_history['val'].append(val_metrics)
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.6f}, "
                f"Val Loss: {val_metrics['total_loss']:.6f}"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['total_loss'], epoch)
                self.writer.add_scalar('epoch/val_loss', val_metrics['total_loss'], epoch)
                self.writer.add_scalar('epoch/on_target_loss', train_metrics['on_target_loss'], epoch)
                self.writer.add_scalar('epoch/off_target_loss', train_metrics['off_target_loss'], epoch)
            
            # WandB logging
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'on_target_loss': train_metrics['on_target_loss'],
                    'off_target_loss': train_metrics['off_target_loss']
                })
            except:
                pass
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint if best model
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.validation.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info("Training completed")
        
        if self.writer:
            self.writer.close()
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_encoder_state_dict': self.optimizer_encoder.state_dict(),
            'optimizer_heads_state_dict': self.optimizer_heads.state_dict(),
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'config': self._config_to_dict()
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"Best model saved at epoch {epoch + 1}")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.config.logging.save_frequency == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
        self.optimizer_heads.load_state_dict(checkpoint['optimizer_heads_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def get_training_history(self) -> Dict:
        """Get complete training history"""
        return self.training_history
    
    def save_training_history(self, path: str) -> None:
        """Save training history to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        history_dict = {}
        for split in ['train', 'val']:
            history_dict[split] = []
            for epoch_metrics in self.training_history[split]:
                epoch_dict = {}
                for key, value in epoch_metrics.items():
                    if isinstance(value, torch.Tensor):
                        epoch_dict[key] = value.item()
                    else:
                        epoch_dict[key] = value
                history_dict[split].append(epoch_dict)
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {path}")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("TRAINER TESTING")
    print("=" * 80)
    
    print("\n[INFO] Trainer class created successfully")
    print("[INFO] To use the Trainer:")
    print("  1. Load configuration: config_loader = ConfigLoader('configs/model_config.yaml')")
    print("  2. Create dataloaders: dataloaders = create_dataloaders(config)")
    print("  3. Initialize model: model = CRISPRUniPredict(device='cuda')")
    print("  4. Create trainer: trainer = Trainer(model, dataloaders, config)")
    print("  5. Train: history = trainer.train()")
    print("\n" + "=" * 80)
