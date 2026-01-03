"""
Configuration management for LRD-PEFT experiments.
Ensures reproducibility and easy hyperparameter tuning.
"""

import os
import json
import torch
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class ModelConfig:
    """Model architecture and loading configurations."""
    
    # Teacher model
    teacher_model_name: str = "meta-llama/Llama-2-7b-hf"
    teacher_load_in_4bit: bool = True
    
    # Student model
    student_model_name: str = "meta-llama/Llama-2-7b-hf"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Layer configuration
    num_layers: int = 32
    hidden_dim: int = 4096
    distill_layers: List[int] = field(default_factory=lambda: [8, 9, 10, 11, 12])
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "float16"  # or "bfloat16"


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings."""
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"  # "linear", "cosine", "constant"
    
    # Training loop
    num_epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Loss weights
    alpha_distill: float = 0.1  # Weight for distillation loss
    distill_loss_type: str = "mse"  # "mse", "cosine", "kl"
    temperature: float = 1.0  # Temperature for KL divergence
    
    # Logging and checkpointing
    logging_steps: int = 50
    eval_steps: int = 375
    save_steps: int = 375
    save_total_limit: int = 3  # Maximum number of checkpoints to keep
    
    # Mixed precision training
    fp16: bool = True
    bf16: bool = False
    
    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""
    
    # Dataset
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    
    # Splits
    train_split: str = "train"
    val_split: str = "test"
    test_split: str = "test"
    
    # Sampling
    num_train_samples: Optional[int] = None  # None = use all
    num_val_samples: Optional[int] = None
    num_test_samples: Optional[int] = None
    
    # Preprocessing
    max_length: int = 512
    truncation_side: str = "left"  # Preserve answer at the end
    padding: str = "max_length"
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class ExperimentConfig:
    """Experiment tracking and output settings."""
    
    # Experiment identification
    experiment_name: str = "lrd-peft"
    run_name: Optional[str] = None  # Auto-generated if None
    description: str = "Latent Reasoning Distillation with PEFT"
    
    # Output directories
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # Logging backends
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: Optional[str] = "lrd-peft"
    wandb_entity: Optional[str] = None
    
    # Reproducibility
    save_config: bool = True
    log_system_info: bool = True
    
    # Evaluation
    evaluate_before_training: bool = False
    evaluate_after_training: bool = True
    
    # Auto-generated fields
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


@dataclass
class LRDPEFTConfig:
    """Complete configuration for LRD-PEFT training."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Auto-generate run name if not provided
        if self.experiment.run_name is None:
            self.experiment.run_name = (
                f"{self.experiment.experiment_name}_"
                f"alpha{self.training.alpha_distill}_"
                f"r{self.model.lora_r}_"
                f"{self.experiment.timestamp}"
            )
        
        # Create output directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.experiment.output_dir,
            self.experiment.logging_dir,
            self.experiment.cache_dir
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "experiment": asdict(self.experiment)
        }
    
    def save(self, path: Optional[str] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = os.path.join(
                self.experiment.output_dir,
                f"config_{self.experiment.timestamp}.json"
            )
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Configuration saved to: {path}")
        return path
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LRDPEFTConfig":
        """Load configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {}))
        )
    
    @classmethod
    def load(cls, path: str) -> "LRDPEFTConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*70)
        print("LRD-PEFT Configuration Summary")
        print("="*70)
        
        print(f"\n📋 Experiment: {self.experiment.run_name}")
        print(f"   Description: {self.experiment.description}")
        print(f"   Output: {self.experiment.output_dir}")
        
        print(f"\n🤖 Models:")
        print(f"   Teacher: {self.model.teacher_model_name}")
        print(f"   Student: {self.model.student_model_name}")
        print(f"   LoRA rank: {self.model.lora_r} (alpha={self.model.lora_alpha})")
        print(f"   Distill layers: {self.model.distill_layers}")
        
        print(f"\n⚙️  Training:")
        print(f"   Epochs: {self.training.num_epochs}")
        print(f"   Batch size: {self.training.train_batch_size} "
              f"(effective={self.training.train_batch_size * self.training.gradient_accumulation_steps})")
        print(f"   Learning rate: {self.training.learning_rate}")
        print(f"   Alpha (distill): {self.training.alpha_distill}")
        print(f"   Warmup ratio: {self.training.warmup_ratio}")
        
        print(f"\n📊 Data:")
        print(f"   Dataset: {self.data.dataset_name}")
        print(f"   Max length: {self.data.max_length}")
        print(f"   Train samples: {self.data.num_train_samples or 'all'}")
        print(f"   Val samples: {self.data.num_val_samples or 'all'}")
        
        print(f"\n🔧 System:")
        print(f"   Device: {self.model.device}")
        print(f"   Precision: {'FP16' if self.training.fp16 else 'FP32'}")
        print(f"   Seed: {self.training.seed}")
        
        print("="*70 + "\n")


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"🔒 Random seed set to: {seed}")


def get_system_info() -> Dict[str, Any]:
    """
    Gather system information for reproducibility.
    
    Returns:
        Dictionary with system info
    """
    info = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "num_gpus": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory": [f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB" 
                          for i in range(torch.cuda.device_count())]
        })
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    
    print("\n" + "="*70)
    print("System Information")
    print("="*70)
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA version: {info['cuda_version']}")
        print(f"cuDNN version: {info['cudnn_version']}")
        print(f"GPUs: {info['num_gpus']}")
        for i, (name, mem) in enumerate(zip(info['gpu_names'], info['gpu_memory'])):
            print(f"  GPU {i}: {name} ({mem})")
    
    print("="*70 + "\n")


# Default configuration instance
default_config = LRDPEFTConfig()


if __name__ == "__main__":
    # Example usage
    config = LRDPEFTConfig()
    config.print_summary()
    print_system_info()
    
    # Save config
    config.save()
    
    # Set seed
    set_seed(config.training.seed)
