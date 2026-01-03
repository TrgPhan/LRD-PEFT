"""
Main training script for LRD-PEFT.
Integrates all components with comprehensive logging and error handling.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local modules
from src.config import LRDPEFTConfig, set_seed, print_system_info, get_system_info
from src.model import TeacherModel, StudentModel, print_model_info
from src.distillation import CombinedDistillationLoss
from src.lora import create_lora_model
from src.trainer import LRDPEFTTrainer
from src.utils import create_dataloaders, evaluate_accuracy
from src.metrics import MetricsTracker
from src.experiment_tracker import ExperimentTracker
from src.visualization import VisualizationHelper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LRD-PEFT model")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON file")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    
    # Model
    parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Teacher model name")
    parser.add_argument("--student_model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Student model name")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Distillation weight")
    
    # Data
    parser.add_argument("--num_train_samples", type=int, default=None,
                       help="Number of training samples (None for all)")
    parser.add_argument("--num_val_samples", type=int, default=None,
                       help="Number of validation samples")
    
    # Experiment
    parser.add_argument("--experiment_name", type=str, default="lrd-peft",
                       help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="lrd-peft",
                       help="W&B project name")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def setup_models(config: LRDPEFTConfig):
    """
    Setup teacher and student models.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (teacher_model, student_model)
    """
    logger.info("="*70)
    logger.info("Setting up models")
    logger.info("="*70)
    
    # Teacher model
    logger.info("Loading teacher model...")
    teacher = TeacherModel(
        model_name=config.model.teacher_model_name,
        device=config.model.device,
        load_in_4bit=config.model.teacher_load_in_4bit
    )
    print_model_info(teacher, "Teacher Model")
    
    # Student model
    logger.info("Loading student model...")
    student = StudentModel(
        model_name=config.model.student_model_name,
        device=config.model.device,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules
    )
    
    # Apply LoRA to student
    logger.info("Applying LoRA adapters to student...")
    student.model = create_lora_model(
        base_model=student.model,
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=config.model.lora_target_modules
    )
    print_model_info(student, "Student Model")
    
    return teacher, student


def main():
    """Main training function."""
    args = parse_args()
    
    # Load or create config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = LRDPEFTConfig.load(args.config)
    else:
        logger.info("Using default config with CLI overrides")
        config = LRDPEFTConfig()
        
        # Override with CLI arguments
        config.model.teacher_model_name = args.teacher_model
        config.model.student_model_name = args.student_model
        config.model.lora_r = args.lora_r
        config.model.lora_alpha = args.lora_alpha
        config.training.num_epochs = args.num_epochs
        config.training.learning_rate = args.learning_rate
        config.training.train_batch_size = args.batch_size
        config.training.alpha_distill = args.alpha
        config.training.seed = args.seed
        config.data.num_train_samples = args.num_train_samples
        config.data.num_val_samples = args.num_val_samples
        config.experiment.output_dir = args.output_dir
        config.experiment.experiment_name = args.experiment_name
        config.experiment.use_wandb = args.use_wandb
        config.experiment.wandb_project = args.wandb_project
    
    # Print configuration
    config.print_summary()
    print_system_info()
    
    # Set random seed for reproducibility
    set_seed(config.training.seed)
    
    # Save config
    config.save()
    
    # Initialize experiment tracker
    logger.info("Initializing experiment tracker...")
    tracker = ExperimentTracker(
        experiment_name=config.experiment.run_name,
        output_dir=config.experiment.output_dir,
        use_tensorboard=config.experiment.use_tensorboard,
        use_wandb=config.experiment.use_wandb,
        wandb_project=config.experiment.wandb_project,
        wandb_entity=config.experiment.wandb_entity,
        config=config.to_dict()
    )
    
    # Setup models
    try:
        teacher, student = setup_models(config)
    except Exception as e:
        logger.error(f"Failed to setup models: {str(e)}")
        raise
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_dataloader, val_dataloader = create_dataloaders(
            tokenizer=student.tokenizer,
            train_batch_size=config.training.train_batch_size,
            eval_batch_size=config.training.eval_batch_size,
            max_length=config.data.max_length,
            num_train_samples=config.data.num_train_samples,
            num_eval_samples=config.data.num_val_samples,
            num_workers=config.data.num_workers
        )
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {str(e)}")
        raise
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = LRDPEFTTrainer(
        teacher_model=teacher,
        student_model=student,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        alpha=config.training.alpha_distill,
        warmup_ratio=config.training.warmup_ratio,
        output_dir=config.experiment.output_dir,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        device=config.model.device,
        distill_layers=config.model.distill_layers
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("="*70)
    logger.info("Starting training")
    logger.info("="*70)
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        trainer.save_checkpoint("interrupted")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    # Final evaluation
    logger.info("="*70)
    logger.info("Final Evaluation")
    logger.info("="*70)
    
    try:
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_accuracy(
            model=student,
            dataloader=val_dataloader,
            tokenizer=student.tokenizer,
            device=config.model.device
        )
        
        logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        
        # Save metrics
        metrics_tracker.print_summary()
        metrics_path = os.path.join(config.experiment.output_dir, "final_metrics.json")
        metrics_tracker.save(metrics_path)
        
        # Log to experiment tracker
        tracker.save_summary({
            "final_accuracy": test_metrics['accuracy'],
            "final_correct": test_metrics['correct'],
            "final_total": test_metrics['total'],
            "config": config.to_dict(),
            "system_info": get_system_info()
        })
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    try:
        viz = VisualizationHelper(
            output_dir=os.path.join(config.experiment.output_dir, "figures")
        )
        
        # Plot training curves if metrics available
        if trainer.metrics and "train_loss" in trainer.metrics:
            steps = list(range(len(trainer.metrics["train_loss"])))
            viz.plot_training_curves(
                metrics={
                    'total_loss': trainer.metrics["train_loss"],
                    'distill_loss': trainer.metrics.get("train_distill_loss", []),
                    'task_loss': trainer.metrics.get("train_task_loss", [])
                },
                steps=steps
            )
        
        logger.info("Visualizations created")
    except Exception as e:
        logger.warning(f"Failed to create visualizations: {str(e)}")
    
    # Finish experiment tracking
    tracker.finish()
    
    logger.info("="*70)
    logger.info("Experiment Complete!")
    logger.info(f"Results saved to: {config.experiment.output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in main:")
        sys.exit(1)
