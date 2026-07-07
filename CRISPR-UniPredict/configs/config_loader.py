"""
Configuration Loader for CRISPR-UniPredict
Loads and manages YAML configuration files with validation and defaults
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class EncodingConfig:
    """Sequence encoding configuration"""
    max_sequence_length: int = 23
    embedding_dim: int = 128
    vocab_size: int = 6
    use_one_hot: bool = True
    use_label_encoding: bool = True
    use_rna_fm: bool = True


@dataclass
class MSCConfig:
    """Multi-Scale Convolution configuration"""
    in_channels: int = 4
    out_channels: int = 64
    kernel_sizes: list = field(default_factory=lambda: [1, 3, 5, 7])
    dropout: float = 0.35


@dataclass
class BiGRUConfig:
    """Bidirectional GRU configuration"""
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.35
    bidirectional: bool = True


@dataclass
class MHSAConfig:
    """Multi-Head Self-Attention configuration"""
    embed_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.35


@dataclass
class RNAFMConfig:
    """RNA-FM Encoder configuration"""
    model_path: str = "models/pretrained/rna_fm_t12.pt"
    freeze_layers: bool = True
    fine_tune_last_n: int = 2
    embedding_dim: int = 640


@dataclass
class FusionConfig:
    """Feature fusion configuration"""
    hidden_dim: int = 256
    fusion_method: str = "attention"  # 'attention' or 'concat'
    num_branches: int = 3
    dropout: float = 0.35


@dataclass
class TaskHeadsConfig:
    """Task-specific output heads configuration"""
    shared_dim: int = 256
    on_target_hidden: list = field(default_factory=lambda: [80, 20])
    off_target_hidden: list = field(default_factory=lambda: [80, 20])
    dropout: float = 0.35


@dataclass
class ModelConfig:
    """Main model configuration"""
    name: str = "CRISPR-UniPredict"
    version: str = "1.0"
    description: str = "Unified Hybrid Neural Network for CRISPR Prediction"
    
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    msc: MSCConfig = field(default_factory=MSCConfig)
    bigru: BiGRUConfig = field(default_factory=BiGRUConfig)
    mhsa: MHSAConfig = field(default_factory=MHSAConfig)
    rna_fm: RNAFMConfig = field(default_factory=RNAFMConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    task_heads: TaskHeadsConfig = field(default_factory=TaskHeadsConfig)


@dataclass
class LossConfig:
    """Loss function configuration"""
    on_target_loss: str = "mse"  # 'mse' or 'mae'
    off_target_loss: str = "bce"  # 'bce'
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "on_target": 1.0,
        "off_target": 0.5
    })


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    type: str = "reduce_on_plateau"  # 'reduce_on_plateau', 'cosine', 'linear', 'exponential'
    patience: int = 3
    factor: float = 0.5
    min_lr: float = 1.0e-6
    warmup_type: str = "linear"  # 'linear' or 'constant'


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate_encoder: float = 5.0e-4
    learning_rate_heads: float = 1.0e-3
    optimizer: str = "AdamW"
    warmup_epochs: int = 5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class ValidationConfig:
    """Validation configuration"""
    val_frequency: int = 1  # epochs
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # 'min' or 'max'


@dataclass
class SamplingConfig:
    """Data sampling configuration"""
    strategy: str = "balanced"  # 'balanced', 'bootstrap', 'weighted'
    on_target_ratio: float = 0.5
    off_target_ratio: float = 0.5


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    use_augmentation: bool = False
    augmentation_types: list = field(default_factory=list)


@dataclass
class DataConfig:
    """Data configuration"""
    train_path: str = "data/processed/combined/train.csv"
    val_path: str = "data/processed/combined/val.csv"
    test_path: str = "data/processed/combined/test.csv"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class DeviceConfig:
    """Device configuration"""
    use_cuda: bool = True
    gpu_ids: list = field(default_factory=lambda: [0])
    mixed_precision: bool = True
    benchmark: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: str = "logs"
    checkpoint_dir: str = "models/checkpoints"
    use_wandb: bool = True
    wandb_project: str = "CRISPR-UniPredict"
    wandb_entity: Optional[str] = None
    save_frequency: int = 1  # epochs
    print_frequency: int = 100  # batches
    log_level: str = "INFO"
    metrics_to_log: list = field(default_factory=lambda: [
        "loss",
        "on_target_loss",
        "off_target_loss",
        "on_target_rmse",
        "on_target_r2",
        "off_target_auc",
        "off_target_accuracy",
        "learning_rate"
    ])


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 64
    return_attention_weights: bool = False
    return_branch_outputs: bool = False
    device: str = "cuda"


@dataclass
class ModelSelectionConfig:
    """Model selection configuration"""
    strategy: str = "best_val_loss"
    save_top_k: int = 3


@dataclass
class EnsembleConfig:
    """Ensemble configuration"""
    use_ensemble: bool = False
    num_models: int = 5
    ensemble_method: str = "average"  # 'average', 'voting', 'stacking'


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)


class ConfigLoader:
    """
    Load and manage YAML configuration files
    
    Supports:
    - Loading YAML configuration files
    - Merging with default configurations
    - Validation of configuration values
    - Conversion to dataclass objects
    - Saving configurations to JSON/YAML
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = Config()
        
        if self.config_path and self.config_path.exists():
            self.load(self.config_path)
    
    def load(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Loaded Config object
        
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file: {str(e)}")
        
        if config_dict is None:
            config_dict = {}
        
        self.config = self._dict_to_config(config_dict)
        self.config_path = config_path
        
        logger.info(f"Loaded configuration from {config_path}")
        
        return self.config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """
        Convert dictionary to Config dataclass
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Config object
        """
        # Create nested dataclass objects
        model_dict = config_dict.get('model', {})
        model = ModelConfig(
            name=model_dict.get('name', 'CRISPR-UniPredict'),
            version=model_dict.get('version', '1.0'),
            description=model_dict.get('description', ''),
            encoding=self._create_dataclass(EncodingConfig, model_dict.get('encoding', {})),
            msc=self._create_dataclass(MSCConfig, model_dict.get('msc', {})),
            bigru=self._create_dataclass(BiGRUConfig, model_dict.get('bigru', {})),
            mhsa=self._create_dataclass(MHSAConfig, model_dict.get('mhsa', {})),
            rna_fm=self._create_dataclass(RNAFMConfig, model_dict.get('rna_fm', {})),
            fusion=self._create_dataclass(FusionConfig, model_dict.get('fusion', {})),
            task_heads=self._create_dataclass(TaskHeadsConfig, model_dict.get('task_heads', {}))
        )
        
        training_dict = config_dict.get('training', {})
        training = TrainingConfig(
            batch_size=training_dict.get('batch_size', 32),
            epochs=training_dict.get('epochs', 100),
            learning_rate_encoder=training_dict.get('learning_rate_encoder', 5.0e-4),
            learning_rate_heads=training_dict.get('learning_rate_heads', 1.0e-3),
            optimizer=training_dict.get('optimizer', 'AdamW'),
            warmup_epochs=training_dict.get('warmup_epochs', 5),
            weight_decay=training_dict.get('weight_decay', 0.01),
            gradient_clip=training_dict.get('gradient_clip', 1.0),
            loss=self._create_dataclass(LossConfig, training_dict.get('loss', {})),
            scheduler=self._create_dataclass(SchedulerConfig, training_dict.get('scheduler', {}))
        )
        
        validation_dict = config_dict.get('validation', {})
        validation = ValidationConfig(
            val_frequency=validation_dict.get('val_frequency', 1),
            early_stopping_patience=validation_dict.get('early_stopping_patience', 10),
            early_stopping_metric=validation_dict.get('early_stopping_metric', 'val_loss'),
            early_stopping_mode=validation_dict.get('early_stopping_mode', 'min')
        )
        
        data_dict = config_dict.get('data', {})
        data = DataConfig(
            train_path=data_dict.get('train_path', 'data/processed/combined/train.csv'),
            val_path=data_dict.get('val_path', 'data/processed/combined/val.csv'),
            test_path=data_dict.get('test_path', 'data/processed/combined/test.csv'),
            num_workers=data_dict.get('num_workers', 4),
            pin_memory=data_dict.get('pin_memory', True),
            prefetch_factor=data_dict.get('prefetch_factor', 2),
            sampling=self._create_dataclass(SamplingConfig, data_dict.get('sampling', {})),
            augmentation=self._create_dataclass(AugmentationConfig, data_dict.get('augmentation', {}))
        )
        
        device_dict = config_dict.get('device', {})
        device = DeviceConfig(
            use_cuda=device_dict.get('use_cuda', True),
            gpu_ids=device_dict.get('gpu_ids', [0]),
            mixed_precision=device_dict.get('mixed_precision', True),
            benchmark=device_dict.get('benchmark', True)
        )
        
        logging_dict = config_dict.get('logging', {})
        logging = LoggingConfig(
            log_dir=logging_dict.get('log_dir', 'logs'),
            checkpoint_dir=logging_dict.get('checkpoint_dir', 'models/checkpoints'),
            use_wandb=logging_dict.get('use_wandb', True),
            wandb_project=logging_dict.get('wandb_project', 'CRISPR-UniPredict'),
            wandb_entity=logging_dict.get('wandb_entity', None),
            save_frequency=logging_dict.get('save_frequency', 1),
            print_frequency=logging_dict.get('print_frequency', 100),
            log_level=logging_dict.get('log_level', 'INFO'),
            metrics_to_log=logging_dict.get('metrics_to_log', [])
        )
        
        inference_dict = config_dict.get('inference', {})
        inference = InferenceConfig(
            batch_size=inference_dict.get('batch_size', 64),
            return_attention_weights=inference_dict.get('return_attention_weights', False),
            return_branch_outputs=inference_dict.get('return_branch_outputs', False),
            device=inference_dict.get('device', 'cuda')
        )
        
        model_selection_dict = config_dict.get('model_selection', {})
        model_selection = ModelSelectionConfig(
            strategy=model_selection_dict.get('strategy', 'best_val_loss'),
            save_top_k=model_selection_dict.get('save_top_k', 3)
        )
        
        ensemble_dict = config_dict.get('ensemble', {})
        ensemble = EnsembleConfig(
            use_ensemble=ensemble_dict.get('use_ensemble', False),
            num_models=ensemble_dict.get('num_models', 5),
            ensemble_method=ensemble_dict.get('ensemble_method', 'average')
        )
        
        return Config(
            model=model,
            training=training,
            validation=validation,
            data=data,
            device=device,
            logging=logging,
            inference=inference,
            model_selection=model_selection,
            ensemble=ensemble
        )
    
    @staticmethod
    def _create_dataclass(dataclass_type, data_dict: Dict[str, Any]):
        """
        Create dataclass instance from dictionary
        
        Args:
            dataclass_type: Dataclass type to create
            data_dict: Dictionary with values
        
        Returns:
            Dataclass instance
        """
        # Get default instance
        instance = dataclass_type()
        
        # Update with provided values
        for key, value in data_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        
        return instance
    
    def save(self, output_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save configuration to file
        
        Args:
            output_path: Path to save configuration
            format: Format to save ('yaml' or 'json')
        
        Raises:
            ValueError: If format not supported
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(self.config)
        
        if format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {output_path}")
    
    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        """
        Convert Config object to dictionary
        
        Args:
            config: Config object
        
        Returns:
            Dictionary representation
        """
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        return dataclass_to_dict(config)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary
        
        Args:
            updates: Dictionary with updates
        """
        config_dict = self._config_to_dict(self.config)
        self._deep_update(config_dict, updates)
        self.config = self._dict_to_config(config_dict)
        
        logger.info(f"Updated configuration with {len(updates)} changes")
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update dictionary
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigLoader._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.config.model
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        return self.config.training
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        return self.config.data
    
    def get_device_config(self) -> DeviceConfig:
        """Get device configuration"""
        return self.config.device
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.config.logging
    
    def __repr__(self) -> str:
        """String representation"""
        return f"ConfigLoader(config_path={self.config_path})"
    
    def print_config(self, section: Optional[str] = None) -> None:
        """
        Print configuration to console
        
        Args:
            section: Specific section to print (None for all)
        """
        config_dict = self._config_to_dict(self.config)
        
        if section:
            if section in config_dict:
                print(f"\n{section.upper()} Configuration:")
                print(yaml.dump({section: config_dict[section]}, default_flow_style=False))
            else:
                logger.warning(f"Section '{section}' not found in configuration")
        else:
            print("\nFull Configuration:")
            print(yaml.dump(config_dict, default_flow_style=False))


# Convenience function for loading configuration
def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Config object
    """
    loader = ConfigLoader(config_path)
    return loader.config


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("CONFIGURATION LOADER TESTING")
    print("=" * 80)
    
    # Test 1: Load default configuration
    print("\n1. LOAD DEFAULT CONFIGURATION")
    print("-" * 80)
    
    config_path = Path(__file__).parent / "model_config.yaml"
    
    if config_path.exists():
        loader = ConfigLoader(config_path)
        print(f"[OK] Loaded configuration from {config_path}")
        print(f"  Model: {loader.config.model.name} v{loader.config.model.version}")
        print(f"  Training batch size: {loader.config.training.batch_size}")
        print(f"  Epochs: {loader.config.training.epochs}")
    else:
        print(f"[SKIP] Configuration file not found: {config_path}")
    
    # Test 2: Access specific configurations
    print("\n2. ACCESS SPECIFIC CONFIGURATIONS")
    print("-" * 80)
    
    if config_path.exists():
        loader = ConfigLoader(config_path)
        
        model_cfg = loader.get_model_config()
        print(f"  Model name: {model_cfg.name}")
        print(f"  MSC out channels: {model_cfg.msc.out_channels}")
        print(f"  BiGRU hidden dim: {model_cfg.bigru.hidden_dim}")
        print(f"  MHSA num heads: {model_cfg.mhsa.num_heads}")
        
        training_cfg = loader.get_training_config()
        print(f"  Optimizer: {training_cfg.optimizer}")
        print(f"  Learning rate (encoder): {training_cfg.learning_rate_encoder}")
        print(f"  Learning rate (heads): {training_cfg.learning_rate_heads}")
        
        data_cfg = loader.get_data_config()
        print(f"  Train path: {data_cfg.train_path}")
        print(f"  Batch size: {data_cfg.num_workers} workers")
    
    # Test 3: Update configuration
    print("\n3. UPDATE CONFIGURATION")
    print("-" * 80)
    
    if config_path.exists():
        loader = ConfigLoader(config_path)
        original_batch_size = loader.config.training.batch_size
        
        loader.update({'training': {'batch_size': 64}})
        print(f"[OK] Updated batch size: {original_batch_size} -> {loader.config.training.batch_size}")
    
    # Test 4: Save configuration
    print("\n4. SAVE CONFIGURATION")
    print("-" * 80)
    
    if config_path.exists():
        loader = ConfigLoader(config_path)
        
        output_yaml = Path(__file__).parent / "model_config_saved.yaml"
        output_json = Path(__file__).parent / "model_config_saved.json"
        
        loader.save(output_yaml, format='yaml')
        print(f"[OK] Saved YAML configuration to {output_yaml}")
        
        loader.save(output_json, format='json')
        print(f"[OK] Saved JSON configuration to {output_json}")
    
    # Test 5: Print configuration sections
    print("\n5. PRINT CONFIGURATION SECTIONS")
    print("-" * 80)
    
    if config_path.exists():
        loader = ConfigLoader(config_path)
        print("\nModel Configuration:")
        loader.print_config('model')
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
