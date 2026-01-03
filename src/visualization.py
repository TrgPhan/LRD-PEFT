"""
Visualization utilities for LRD-PEFT analysis.
Generates plots for paper figures and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


class VisualizationHelper:
    """Helper class for creating paper-ready visualizations."""
    
    def __init__(self, output_dir: str = "./figures"):
        """
        Initialize visualization helper.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        steps: List[int],
        save_name: str = "training_curves.pdf"
    ):
        """
        Plot training loss curves (Figure 3a in paper).
        
        Args:
            metrics: Dictionary with keys like 'total_loss', 'distill_loss', 'task_loss'
            steps: List of training steps
            save_name: Filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        if 'total_loss' in metrics:
            ax.plot(steps, metrics['total_loss'], label='Total Loss', linewidth=2)
        if 'distill_loss' in metrics:
            ax.plot(steps, metrics['distill_loss'], label='Distillation Loss', 
                   linewidth=2, linestyle='--')
        if 'task_loss' in metrics:
            ax.plot(steps, metrics['task_loss'], label='Task Loss', 
                   linewidth=2, linestyle=':')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_alpha_sensitivity(
        self,
        alpha_values: List[float],
        accuracies: List[float],
        save_name: str = "alpha_sensitivity.pdf"
    ):
        """
        Plot distillation weight sensitivity (Figure 5 in paper).
        
        Args:
            alpha_values: List of alpha values
            accuracies: Corresponding accuracies
            save_name: Filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        ax.plot(alpha_values, accuracies, marker='o', linewidth=2, markersize=8)
        
        # Highlight optimal alpha
        best_idx = np.argmax(accuracies)
        best_alpha = alpha_values[best_idx]
        best_acc = accuracies[best_idx]
        
        ax.axvline(best_alpha, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal α={best_alpha:.2f}')
        ax.plot(best_alpha, best_acc, 'r*', markersize=15)
        
        ax.set_xlabel('Distillation Weight (α)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Effect of Distillation Weight on Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_layer_similarity(
        self,
        layer_similarities: Dict[int, float],
        save_name: str = "layer_similarity.pdf"
    ):
        """
        Plot layer-wise cosine similarity (Figure 7b in paper).
        
        Args:
            layer_similarities: Dictionary mapping layer index to similarity
            save_name: Filename to save
        """
        layers = sorted(layer_similarities.keys())
        similarities = [layer_similarities[l] for l in layers]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        ax.plot(layers, similarities, marker='o', linewidth=2, markersize=8)
        ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, 
                   label='Good Alignment (>0.8)')
        
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Layer-wise Hidden State Alignment')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_layer_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        student_layers: List[int],
        teacher_layers: List[int],
        save_name: str = "layer_similarity_heatmap.pdf"
    ):
        """
        Plot layer-to-layer similarity heatmap (Figure 9 in paper).
        
        Args:
            similarity_matrix: Matrix of similarities [student_layers, teacher_layers]
            student_layers: List of student layer indices
            teacher_layers: List of teacher layer indices
            save_name: Filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=[f'T{l}' for l in teacher_layers],
            yticklabels=[f'S{l}' for l in student_layers],
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax
        )
        
        ax.set_xlabel('Teacher Layers')
        ax.set_ylabel('Student Layers')
        ax.set_title('Layer-to-Layer Alignment Matrix')
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_efficiency_comparison(
        self,
        methods: List[str],
        accuracies: List[float],
        training_times: List[float],
        parameters: List[int],
        save_name: str = "efficiency_comparison.pdf"
    ):
        """
        Plot efficiency comparison (Figure 6 in paper).
        
        Args:
            methods: List of method names
            accuracies: Accuracy for each method
            training_times: Training time in hours
            parameters: Number of trainable parameters
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training time comparison
        axes[0].barh(methods, training_times, color='skyblue')
        axes[0].set_xlabel('Training Time (hours)')
        axes[0].set_title('Training Efficiency')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Parameter comparison (log scale)
        params_millions = [p / 1e6 for p in parameters]
        axes[1].barh(methods, params_millions, color='lightcoral')
        axes[1].set_xlabel('Trainable Parameters (Millions)')
        axes[1].set_xscale('log')
        axes[1].set_title('Parameter Efficiency')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Accuracy vs Time scatter
        sizes = [np.sqrt(p) for p in params_millions]  # Bubble size by params
        axes[2].scatter(training_times, accuracies, s=[s*50 for s in sizes], 
                       alpha=0.6, c=range(len(methods)), cmap='viridis')
        
        # Add method labels
        for i, method in enumerate(methods):
            axes[2].annotate(method, (training_times[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[2].set_xlabel('Training Time (hours)')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('Accuracy vs Efficiency Trade-off')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_error_breakdown(
        self,
        error_types: Dict[str, int],
        save_name: str = "error_breakdown.pdf"
    ):
        """
        Plot error type breakdown (Table 5 data as pie chart).
        
        Args:
            error_types: Dictionary mapping error type to count
            save_name: Filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        labels = list(error_types.keys())
        sizes = list(error_types.values())
        colors = sns.color_palette('pastel', len(labels))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Improve text
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        ax.set_title('Error Type Distribution')
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_confidence_calibration(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
        save_name: str = "calibration_curve.pdf"
    ):
        """
        Plot calibration curve showing confidence vs accuracy.
        
        Args:
            confidences: Array of confidence scores
            accuracies: Boolean array of correctness
            n_bins: Number of bins
            save_name: Filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax.plot(bin_confidences, bin_accuracies, 'o-', markersize=10, 
               linewidth=2, label='Model Calibration')
        
        # Add count labels
        for x, y, count in zip(bin_confidences, bin_accuracies, bin_counts):
            ax.annotate(f'n={count}', (x, y), xytext=(5, -5),
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_ablation_study(
        self,
        configs: List[str],
        accuracies: List[float],
        baseline_idx: int = 0,
        save_name: str = "ablation_study.pdf"
    ):
        """
        Plot ablation study results (Table 4 as bar chart).
        
        Args:
            configs: List of configuration names
            accuracies: Accuracy for each config
            baseline_idx: Index of full model (baseline)
            save_name: Filename to save
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate deltas from baseline
        baseline_acc = accuracies[baseline_idx]
        deltas = [acc - baseline_acc for acc in accuracies]
        
        # Color based on delta
        colors = ['green' if d >= 0 else 'red' for d in deltas]
        colors[baseline_idx] = 'blue'  # Baseline in blue
        
        # Create horizontal bar chart
        y_pos = np.arange(len(configs))
        ax.barh(y_pos, accuracies, color=colors, alpha=0.7)
        
        # Add accuracy labels
        for i, (acc, delta) in enumerate(zip(accuracies, deltas)):
            label = f'{acc:.1f}%'
            if i != baseline_idx:
                label += f' ({delta:+.1f})'
            ax.text(acc + 0.5, i, label, va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Ablation Study Results')
        ax.axvline(baseline_acc, color='blue', linestyle='--', alpha=0.5,
                  label='Full Model')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def load_metrics_from_jsonl(jsonl_path: str) -> Dict[str, List]:
    """
    Load metrics from JSONL log file.
    
    Args:
        jsonl_path: Path to metrics.jsonl file
        
    Returns:
        Dictionary with lists of metric values
    """
    metrics = {}
    steps = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            
            step = entry.get('step')
            if step is not None:
                steps.append(step)
            
            for key, value in entry.items():
                if key not in ['step', 'timestamp']:
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
    
    metrics['steps'] = steps
    return metrics


if __name__ == "__main__":
    # Example usage
    viz = VisualizationHelper("./figures")
    
    # Example: Training curves
    steps = list(range(0, 1000, 50))
    metrics = {
        'total_loss': [2.0 - 1.5 * (i / 1000) for i in steps],
        'distill_loss': [0.8 - 0.6 * (i / 1000) for i in steps],
        'task_loss': [1.2 - 0.9 * (i / 1000) for i in steps]
    }
    viz.plot_training_curves(metrics, steps)
    
    # Example: Alpha sensitivity
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    accs = [35.7, 41.2, 43.7, 42.1, 40.8, 38.5, 36.2]
    viz.plot_alpha_sensitivity(alphas, accs)
    
    print("Example visualizations created!")
