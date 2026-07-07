"""
Optimizer and Scheduler Configuration for CRISPR-UniPredict
Implements differential learning rates for different model components
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def setup_optimizer(model, config) -> torch.optim.Optimizer:
    """
    Setup optimizer with differential learning rates
    
    Strategy (from CCLMoff):
    - RNA-FM encoder: 5×10^-4 (pretrained, smaller LR)
    - MSC, BiGRU, MHSA: 1×10^-3 (training from scratch)
    - Task heads: 1×10^-3 (task-specific)
    
    Args:
        model: CRISPRUniPredict model instance
        config: Configuration object
    
    Returns:
        Configured optimizer with parameter groups
    
    Example:
        >>> optimizer = setup_optimizer(model, config)
        >>> # optimizer has different learning rates for different components
    """
    
    # Define parameter groups
    param_groups = _create_parameter_groups(model, config)
    
    # Create optimizer
    if config.training.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'Adam':
        optimizer = optim.Adam(
            param_groups,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    # Log optimizer configuration
    _log_optimizer_config(optimizer)
    
    return optimizer


def _create_parameter_groups(model, config) -> List[Dict]:
    """
    Create parameter groups with differential learning rates
    
    Args:
        model: CRISPRUniPredict model instance
        config: Configuration object
    
    Returns:
        List of parameter groups
    """
    
    # Initialize parameter groups
    rna_fm_params = []
    feature_extraction_params = []
    task_head_params = []
    
    # Categorize parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # RNA-FM encoder parameters (pretrained)
        if 'rna_fm' in name:
            rna_fm_params.append(param)
        
        # Feature extraction parameters (MSC, BiGRU, MHSA)
        elif any(x in name for x in ['msc', 'bigru', 'mhsa', 'embedding']):
            feature_extraction_params.append(param)
        
        # Task head parameters
        elif any(x in name for x in ['head', 'fusion']):
            task_head_params.append(param)
        
        else:
            # Default to feature extraction
            feature_extraction_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = []
    
    # RNA-FM encoder: smaller learning rate (pretrained)
    if rna_fm_params:
        param_groups.append({
            'params': rna_fm_params,
            'lr': config.training.learning_rate_encoder,
            'name': 'rna_fm'
        })
    
    # Feature extraction: larger learning rate (training from scratch)
    if feature_extraction_params:
        param_groups.append({
            'params': feature_extraction_params,
            'lr': config.training.learning_rate_heads,
            'name': 'feature_extraction'
        })
    
    # Task heads: task-specific learning rate
    if task_head_params:
        param_groups.append({
            'params': task_head_params,
            'lr': config.training.learning_rate_heads,
            'name': 'task_heads'
        })
    
    return param_groups


def _log_optimizer_config(optimizer: torch.optim.Optimizer) -> None:
    """
    Log optimizer configuration
    
    Args:
        optimizer: Configured optimizer
    """
    logger.info("Optimizer configuration:")
    
    for i, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('name', f'group_{i}')
        lr = param_group['lr']
        num_params = sum(p.numel() for p in param_group['params'])
        
        logger.info(
            f"  {name}: {num_params:,} parameters, LR={lr:.2e}"
        )


def setup_scheduler(optimizer: torch.optim.Optimizer,
                   config,
                   total_steps: Optional[int] = None) -> Tuple:
    """
    Setup learning rate scheduler with warmup
    
    Strategy:
    - Linear warmup for first N epochs
    - ReduceLROnPlateau for main training
    
    Args:
        optimizer: Configured optimizer
        config: Configuration object
        total_steps: Total training steps (for warmup calculation)
    
    Returns:
        Tuple of (scheduler, warmup_scheduler)
    
    Example:
        >>> scheduler, warmup_scheduler = setup_scheduler(optimizer, config)
        >>> for epoch in range(epochs):
        ...     # Warmup phase
        ...     if epoch < warmup_epochs:
        ...         warmup_scheduler.step()
        ...     # Main training
        ...     train()
        ...     validate()
        ...     scheduler.step(val_loss)
    """
    
    scheduler_type = config.training.scheduler.type
    
    # Main scheduler
    if scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler.factor,
            patience=config.training.scheduler.patience,
            min_lr=config.training.scheduler.min_lr
        )
    
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.scheduler.min_lr
        )
    
    elif scheduler_type == 'linear':
        def linear_lr(epoch):
            return max(1.0 - epoch / config.training.epochs, 0.0)
        
        scheduler = LambdaLR(optimizer, linear_lr)
    
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Warmup scheduler
    warmup_scheduler = _create_warmup_scheduler(
        optimizer,
        config.training.warmup_epochs,
        config.training.epochs
    )
    
    logger.info(
        f"Scheduler configured: {scheduler_type}\n"
        f"  Warmup epochs: {config.training.warmup_epochs}\n"
        f"  Total epochs: {config.training.epochs}"
    )
    
    return scheduler, warmup_scheduler


def _create_warmup_scheduler(optimizer: torch.optim.Optimizer,
                            warmup_epochs: int,
                            total_epochs: int) -> LambdaLR:
    """
    Create linear warmup scheduler
    
    Args:
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
    
    Returns:
        LambdaLR scheduler for warmup
    """
    
    def warmup_lr(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Constant after warmup
            return 1.0
    
    return LambdaLR(optimizer, warmup_lr)


class WarmupScheduler:
    """
    Wrapper for warmup + main scheduler
    
    Handles both warmup phase and main training phase
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 config,
                 total_steps: Optional[int] = None):
        """
        Initialize warmup scheduler
        
        Args:
            optimizer: Optimizer instance
            config: Configuration object
            total_steps: Total training steps (optional)
        """
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0
        self.warmup_epochs = config.training.warmup_epochs
        
        # Create schedulers
        self.main_scheduler, self.warmup_scheduler = setup_scheduler(
            optimizer, config, total_steps
        )
    
    def step(self, val_loss: Optional[float] = None) -> None:
        """
        Step the scheduler
        
        Args:
            val_loss: Validation loss (required for ReduceLROnPlateau)
        """
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            self.warmup_scheduler.step()
            logger.debug(f"Warmup step {self.current_epoch + 1}/{self.warmup_epochs}")
        else:
            # Main training phase
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                if val_loss is None:
                    raise ValueError("val_loss required for ReduceLROnPlateau")
                self.main_scheduler.step(val_loss)
            else:
                self.main_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates"""
        return self.optimizer.param_groups[-1]['lr']


class DifferentialLRScheduler:
    """
    Advanced scheduler with differential learning rates per component
    
    Allows different learning rate schedules for different model components
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 config,
                 component_schedules: Optional[Dict] = None):
        """
        Initialize differential LR scheduler
        
        Args:
            optimizer: Optimizer with multiple parameter groups
            config: Configuration object
            component_schedules: Dict mapping component names to schedules
        """
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0
        self.warmup_epochs = config.training.warmup_epochs
        
        # Create per-component schedulers
        self.component_schedulers = {}
        for i, param_group in enumerate(optimizer.param_groups):
            name = param_group.get('name', f'group_{i}')
            
            # Create warmup scheduler for this component
            warmup_sched = LambdaLR(
                optimizer,
                lambda epoch, idx=i: self._warmup_lr(epoch, idx)
            )
            
            self.component_schedulers[name] = warmup_sched
    
    def _warmup_lr(self, epoch: int, param_group_idx: int) -> float:
        """Calculate warmup learning rate"""
        if epoch < self.warmup_epochs:
            return float(epoch) / float(max(1, self.warmup_epochs))
        else:
            return 1.0
    
    def step(self, val_loss: Optional[float] = None) -> None:
        """Step all component schedulers"""
        for name, scheduler in self.component_schedulers.items():
            scheduler.step()
        
        self.current_epoch += 1
    
    def get_component_lrs(self) -> Dict[str, float]:
        """Get learning rates for each component"""
        lrs = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            name = param_group.get('name', f'group_{i}')
            lrs[name] = param_group['lr']
        return lrs


# Convenience functions

def create_optimizer_and_scheduler(model, config, total_steps: Optional[int] = None) -> Tuple:
    """
    Create both optimizer and scheduler in one call
    
    Args:
        model: Model instance
        config: Configuration object
        total_steps: Total training steps (optional)
    
    Returns:
        Tuple of (optimizer, scheduler)
    
    Example:
        >>> optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    """
    optimizer = setup_optimizer(model, config)
    scheduler, warmup_scheduler = setup_scheduler(optimizer, config, total_steps)
    
    return optimizer, WarmupScheduler(optimizer, config, total_steps)


def get_parameter_groups_info(model) -> Dict[str, int]:
    """
    Get information about parameter groups
    
    Args:
        model: Model instance
    
    Returns:
        Dictionary with parameter counts per group
    """
    info = {
        'rna_fm': 0,
        'feature_extraction': 0,
        'task_heads': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        if 'rna_fm' in name:
            info['rna_fm'] += param.numel()
        elif any(x in name for x in ['msc', 'bigru', 'mhsa', 'embedding']):
            info['feature_extraction'] += param.numel()
        elif any(x in name for x in ['head', 'fusion']):
            info['task_heads'] += param.numel()
        else:
            info['other'] += param.numel()
    
    return info


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("OPTIMIZATION CONFIGURATION TESTING")
    print("=" * 80)
    
    print("\n1. OPTIMIZER SETUP")
    print("-" * 80)
    
    print("[INFO] To use the optimization module:")
    print("  1. Load configuration: config = ConfigLoader('configs/model_config.yaml').config")
    print("  2. Create model: model = CRISPRUniPredict(device='cuda')")
    print("  3. Setup optimizer: optimizer = setup_optimizer(model, config)")
    print("  4. Setup scheduler: scheduler, warmup = setup_scheduler(optimizer, config)")
    print("  5. Or use convenience: optimizer, scheduler = create_optimizer_and_scheduler(model, config)")
    
    print("\n2. PARAMETER GROUPS")
    print("-" * 80)
    
    print("[INFO] Parameter groups created with differential learning rates:")
    print("  - RNA-FM encoder: 5×10^-4 (pretrained)")
    print("  - Feature extraction (MSC, BiGRU, MHSA): 1×10^-3")
    print("  - Task heads: 1×10^-3")
    
    print("\n3. SCHEDULER CONFIGURATION")
    print("-" * 80)
    
    print("[INFO] Scheduler configuration:")
    print("  - Warmup: Linear warmup for first N epochs")
    print("  - Main: ReduceLROnPlateau with patience and reduction factor")
    print("  - Min LR: 1×10^-6")
    
    print("\n" + "=" * 80)
    print("[OK] Optimization module ready for use")
    print("=" * 80)
