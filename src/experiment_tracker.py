"""
Experiment tracking and logging for LRD-PEFT.
Supports TensorBoard, Weights & Biases, and custom JSON logging.
"""

import os
import json
import torch
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save logs and checkpoints
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
            config: Configuration dictionary to log
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize backends
        self.tensorboard_writer = None
        self.wandb_run = None
        
        if use_tensorboard:
            self._init_tensorboard()
        
        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity)
        
        # JSON logging
        self.json_log_path = os.path.join(output_dir, "metrics.jsonl")
        self.summary_path = os.path.join(output_dir, "summary.json")
        
        # Save config
        if config:
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Config saved to: {config_path}")
        
        logger.info(f"Experiment tracker initialized: {experiment_name}")
        logger.info(f"Output directory: {output_dir}")
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = os.path.join(self.output_dir, "tensorboard")
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
            logger.info(f"  Run: tensorboard --logdir={log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tensorboard_writer = None
    
    def _init_wandb(self, project: Optional[str], entity: Optional[str]):
        """Initialize Weights & Biases."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=project or "lrd-peft",
                entity=entity,
                name=self.experiment_name,
                config=self.config,
                dir=self.output_dir
            )
            logger.info(f"W&B run initialized: {self.wandb_run.url}")
        except ImportError:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
            self.wandb_run = None
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {str(e)}")
            self.wandb_run = None
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/iteration
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        # Add prefix to metric names
        prefixed_metrics = {
            f"{prefix}{k}" if prefix else k: v
            for k, v in metrics.items()
        }
        
        # TensorBoard
        if self.tensorboard_writer:
            for name, value in prefixed_metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Weights & Biases
        if self.wandb_run:
            self.wandb_run.log(prefixed_metrics, step=step)
        
        # JSON logging (append mode)
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **prefixed_metrics
        }
        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        if self.tensorboard_writer:
            # TensorBoard expects string values for hparams
            hparams_str = {k: str(v) for k, v in hparams.items()}
            self.tensorboard_writer.add_hparams(hparams_str, {})
        
        if self.wandb_run:
            self.wandb_run.config.update(hparams)
    
    def log_model_graph(self, model: torch.nn.Module, input_data: torch.Tensor):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_data: Sample input tensor
        """
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_data)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {str(e)}")
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """
        Log text data.
        
        Args:
            tag: Tag/name for the text
            text: Text content
            step: Step number
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_text(tag, text, step)
        
        if self.wandb_run:
            self.wandb_run.log({tag: text}, step=step)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            step: Current training step
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint data
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {}
        }
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save as "best" if specified
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        # Save as "latest" (for easy resuming)
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> int:
        """
        Load checkpoint and resume training.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to restore state
            scheduler: Scheduler to restore state
            checkpoint_path: Path to checkpoint (if None, loads latest)
            
        Returns:
            Step number from checkpoint
        """
        if checkpoint_path is None:
            # Load latest checkpoint
            checkpoint_path = os.path.join(self.output_dir, "checkpoints", "latest.pt")
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Restore model
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore optimizer
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore scheduler
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        step = checkpoint.get("step", 0)
        logger.info(f"Resumed from step {step}")
        
        return step
    
    def save_summary(self, summary: Dict[str, Any]):
        """
        Save final experiment summary.
        
        Args:
            summary: Dictionary with experiment results
        """
        summary["experiment_name"] = self.experiment_name
        summary["timestamp"] = datetime.now().isoformat()
        
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment summary saved: {self.summary_path}")
        
        # Log to W&B
        if self.wandb_run:
            self.wandb_run.summary.update(summary)
    
    def finish(self):
        """Clean up and finish tracking."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("W&B run finished")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class CheckpointManager:
    """
    Manage checkpoints with automatic cleanup.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_best: bool = True,
        metric_name: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            metric_name: Metric to use for "best" determination
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.checkpoints = []  # List of (step, path, metric) tuples
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        self.best_checkpoint = None
    
    def save(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save checkpoint with automatic cleanup.
        
        Args:
            state_dict: State dictionary to save
            step: Training step
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
        
        # Save checkpoint
        torch.save(state_dict, checkpoint_path)
        
        # Track checkpoint
        metric_value = metrics.get(self.metric_name) if metrics else None
        self.checkpoints.append((step, checkpoint_path, metric_value))
        
        # Check if best
        if self.save_best and metric_value is not None:
            is_best = (
                (self.mode == "min" and metric_value < self.best_metric) or
                (self.mode == "max" and metric_value > self.best_metric)
            )
            
            if is_best:
                self.best_metric = metric_value
                self.best_checkpoint = checkpoint_path
                
                # Save as best
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(state_dict, best_path)
                logger.info(f"New best checkpoint: {self.metric_name}={metric_value:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return checkpoint_path
    
    def _cleanup(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by step (oldest first)
        self.checkpoints.sort(key=lambda x: x[0])
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            step, path, metric = self.checkpoints.pop(0)
            
            # Don't delete best checkpoint
            if path != self.best_checkpoint and os.path.exists(path):
                os.remove(path)
                logger.debug(f"Removed old checkpoint: {path}")


if __name__ == "__main__":
    # Example usage
    config = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "num_epochs": 3
    }
    
    # Use as context manager
    with ExperimentTracker(
        experiment_name="test_run",
        output_dir="./test_output",
        use_tensorboard=True,
        use_wandb=False,
        config=config
    ) as tracker:
        # Log metrics
        for step in range(100):
            tracker.log_metrics(
                {"loss": 1.0 / (step + 1), "accuracy": step / 100},
                step=step,
                prefix="train/"
            )
        
        # Save summary
        tracker.save_summary({"final_loss": 0.01, "final_accuracy": 0.99})
