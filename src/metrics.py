"""
Comprehensive evaluation metrics for LRD-PEFT.
Includes accuracy, confidence scores, error analysis, and layer-wise metrics.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import re
import json


class MetricsTracker:
    """
    Track and compute comprehensive evaluation metrics.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        self.layer_similarities = defaultdict(list)
        self.layer_losses = defaultdict(list)
        self.error_types = []
        self.difficulties = []  # Optional: problem difficulty ratings
        
    def add_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        confidences: Optional[List[float]] = None,
        layer_similarities: Optional[Dict[int, float]] = None,
        layer_losses: Optional[Dict[int, float]] = None,
        difficulties: Optional[List[str]] = None
    ):
        """
        Add batch results for tracking.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            confidences: List of prediction confidence scores
            layer_similarities: Dictionary of layer-wise cosine similarities
            layer_losses: Dictionary of layer-wise losses
            difficulties: List of problem difficulties ("easy", "medium", "hard")
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        
        if confidences:
            self.confidences.extend(confidences)
        
        if layer_similarities:
            for layer, sim in layer_similarities.items():
                self.layer_similarities[layer].append(sim)
        
        if layer_losses:
            for layer, loss in layer_losses.items():
                self.layer_losses[layer].append(loss)
        
        if difficulties:
            self.difficulties.extend(difficulties)
    
    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute overall and per-difficulty accuracy.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.predictions:
            return {"overall": 0.0}
        
        # Overall accuracy
        correct = sum(
            p.strip() == g.strip() 
            for p, g in zip(self.predictions, self.ground_truths)
        )
        total = len(self.predictions)
        overall_acc = correct / total
        
        metrics = {
            "overall": overall_acc,
            "correct": correct,
            "total": total
        }
        
        # Per-difficulty accuracy
        if self.difficulties:
            for diff in ["easy", "medium", "hard"]:
                indices = [i for i, d in enumerate(self.difficulties) if d == diff]
                if indices:
                    correct_diff = sum(
                        self.predictions[i].strip() == self.ground_truths[i].strip()
                        for i in indices
                    )
                    metrics[f"{diff}_accuracy"] = correct_diff / len(indices)
                    metrics[f"{diff}_count"] = len(indices)
        
        return metrics
    
    def compute_confidence_metrics(self) -> Dict[str, float]:
        """
        Compute confidence-related metrics.
        
        Returns:
            Dictionary with confidence metrics
        """
        if not self.confidences:
            return {}
        
        confidences = np.array(self.confidences)
        correct_mask = np.array([
            p.strip() == g.strip()
            for p, g in zip(self.predictions, self.ground_truths)
        ])
        
        metrics = {
            "avg_confidence": float(confidences.mean()),
            "std_confidence": float(confidences.std()),
            "min_confidence": float(confidences.min()),
            "max_confidence": float(confidences.max()),
        }
        
        # Confidence on correct vs incorrect predictions
        if correct_mask.sum() > 0:
            metrics["avg_confidence_correct"] = float(confidences[correct_mask].mean())
        if (~correct_mask).sum() > 0:
            metrics["avg_confidence_incorrect"] = float(confidences[~correct_mask].mean())
        
        # Calibration: compute Expected Calibration Error (ECE)
        metrics["ece"] = self._compute_ece(confidences, correct_mask)
        
        return metrics
    
    def _compute_ece(
        self,
        confidences: np.ndarray,
        correct_mask: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        Measures the difference between confidence and accuracy.
        
        Args:
            confidences: Array of confidence scores [0, 1]
            correct_mask: Boolean array indicating correctness
            n_bins: Number of bins for calibration
            
        Returns:
            ECE score (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            # Find predictions in this confidence bin
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                # Average confidence in bin
                avg_confidence = confidences[in_bin].mean()
                # Accuracy in bin
                avg_accuracy = correct_mask[in_bin].mean()
                # Weighted difference
                ece += (in_bin.sum() / len(confidences)) * abs(avg_confidence - avg_accuracy)
        
        return float(ece)
    
    def compute_layer_metrics(self) -> Dict[str, Any]:
        """
        Compute layer-wise metrics.
        
        Returns:
            Dictionary with layer-wise statistics
        """
        metrics = {}
        
        # Layer-wise similarities
        if self.layer_similarities:
            metrics["layer_similarities"] = {}
            for layer, sims in self.layer_similarities.items():
                metrics["layer_similarities"][f"layer_{layer}"] = {
                    "mean": float(np.mean(sims)),
                    "std": float(np.std(sims)),
                    "min": float(np.min(sims)),
                    "max": float(np.max(sims))
                }
            
            # Average across all layers
            all_sims = [s for sims in self.layer_similarities.values() for s in sims]
            metrics["avg_layer_similarity"] = float(np.mean(all_sims))
        
        # Layer-wise losses
        if self.layer_losses:
            metrics["layer_losses"] = {}
            for layer, losses in self.layer_losses.items():
                metrics["layer_losses"][f"layer_{layer}"] = {
                    "mean": float(np.mean(losses)),
                    "std": float(np.std(losses)),
                    "min": float(np.min(losses)),
                    "max": float(np.max(losses))
                }
            
            # Average across all layers
            all_losses = [l for losses in self.layer_losses.values() for l in losses]
            metrics["avg_layer_loss"] = float(np.mean(all_losses))
        
        return metrics
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Perform detailed error analysis.
        
        Categorizes errors into:
        - Calculation errors
        - Reasoning step missing
        - Misunderstanding
        - Correct method, wrong execution
        - Other
        
        Returns:
            Dictionary with error statistics
        """
        errors = defaultdict(int)
        total_errors = 0
        error_examples = defaultdict(list)
        
        for pred, truth in zip(self.predictions, self.ground_truths):
            if pred.strip() != truth.strip():
                total_errors += 1
                
                # Simple heuristic-based error classification
                error_type = self._classify_error(pred, truth)
                errors[error_type] += 1
                
                # Store a few examples of each error type
                if len(error_examples[error_type]) < 3:
                    error_examples[error_type].append({
                        "prediction": pred,
                        "ground_truth": truth
                    })
        
        # Compute percentages
        error_stats = {}
        if total_errors > 0:
            for error_type, count in errors.items():
                error_stats[error_type] = {
                    "count": count,
                    "percentage": count / total_errors * 100
                }
        
        return {
            "total_errors": total_errors,
            "error_rate": total_errors / len(self.predictions) if self.predictions else 0.0,
            "error_breakdown": error_stats,
            "error_examples": dict(error_examples)
        }
    
    def _classify_error(self, prediction: str, ground_truth: str) -> str:
        """
        Classify error type based on prediction and ground truth.
        
        This is a simple heuristic classification. In practice, you might
        want to use more sophisticated methods or manual annotation.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Error type string
        """
        # Extract numbers
        pred_numbers = re.findall(r'\d+\.?\d*', prediction)
        truth_numbers = re.findall(r'\d+\.?\d*', ground_truth)
        
        # No numbers in prediction
        if not pred_numbers:
            return "reasoning_step_missing"
        
        # Numbers present but different
        if pred_numbers and truth_numbers:
            try:
                pred_val = float(pred_numbers[-1])
                truth_val = float(truth_numbers[-1])
                
                # Check if it's a simple calculation error (e.g., off by small amount)
                if abs(pred_val - truth_val) / max(abs(truth_val), 1) < 0.5:
                    return "calculation_error"
                else:
                    return "misunderstanding"
            except ValueError:
                return "other"
        
        return "other"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete summary of all metrics.
        
        Returns:
            Dictionary with all computed metrics
        """
        summary = {}
        
        # Accuracy metrics
        summary["accuracy"] = self.compute_accuracy()
        
        # Confidence metrics
        if self.confidences:
            summary["confidence"] = self.compute_confidence_metrics()
        
        # Layer metrics
        if self.layer_similarities or self.layer_losses:
            summary["layers"] = self.compute_layer_metrics()
        
        # Error analysis
        summary["errors"] = self.analyze_errors()
        
        return summary
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        summary = self.get_summary()
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to: {path}")
    
    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("Evaluation Summary")
        print("="*70)
        
        # Accuracy
        if "accuracy" in summary:
            acc = summary["accuracy"]
            print(f"\n📊 Accuracy:")
            print(f"   Overall: {acc['overall']*100:.2f}% ({acc['correct']}/{acc['total']})")
            
            for key in ["easy", "medium", "hard"]:
                if f"{key}_accuracy" in acc:
                    print(f"   {key.capitalize()}: {acc[f'{key}_accuracy']*100:.2f}% ({acc[f'{key}_count']} samples)")
        
        # Confidence
        if "confidence" in summary:
            conf = summary["confidence"]
            print(f"\n🎯 Confidence:")
            print(f"   Average: {conf['avg_confidence']:.4f}")
            if "avg_confidence_correct" in conf:
                print(f"   Correct predictions: {conf['avg_confidence_correct']:.4f}")
            if "avg_confidence_incorrect" in conf:
                print(f"   Incorrect predictions: {conf['avg_confidence_incorrect']:.4f}")
            print(f"   Calibration (ECE): {conf['ece']:.4f}")
        
        # Layer metrics
        if "layers" in summary:
            layers = summary["layers"]
            if "avg_layer_similarity" in layers:
                print(f"\n🔗 Layer Alignment:")
                print(f"   Average similarity: {layers['avg_layer_similarity']:.4f}")
            if "avg_layer_loss" in layers:
                print(f"   Average loss: {layers['avg_layer_loss']:.4f}")
        
        # Error analysis
        if "errors" in summary:
            errors = summary["errors"]
            print(f"\n❌ Error Analysis:")
            print(f"   Total errors: {errors['total_errors']} ({errors['error_rate']*100:.2f}%)")
            if errors.get("error_breakdown"):
                print(f"   Breakdown:")
                for error_type, stats in errors["error_breakdown"].items():
                    print(f"      {error_type}: {stats['percentage']:.1f}% ({stats['count']})")
        
        print("="*70 + "\n")


def compute_prediction_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence score for predictions.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab]
        
    Returns:
        Confidence scores [batch] (max probability averaged over sequence)
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get max probability at each position
    max_probs, _ = probs.max(dim=-1)  # [batch, seq_len]
    
    # Average over sequence length (excluding padding)
    confidence = max_probs.mean(dim=-1)  # [batch]
    
    return confidence


def compute_layer_similarity_matrix(
    student_hiddens: Dict[int, torch.Tensor],
    teacher_hiddens: Dict[int, torch.Tensor]
) -> np.ndarray:
    """
    Compute full layer-to-layer similarity matrix.
    
    Useful for analyzing which student layers align with which teacher layers.
    
    Args:
        student_hiddens: Student hidden states {layer: tensor}
        teacher_hiddens: Teacher hidden states {layer: tensor}
        
    Returns:
        Similarity matrix [num_student_layers, num_teacher_layers]
    """
    student_layers = sorted(student_hiddens.keys())
    teacher_layers = sorted(teacher_hiddens.keys())
    
    matrix = np.zeros((len(student_layers), len(teacher_layers)))
    
    for i, s_layer in enumerate(student_layers):
        for j, t_layer in enumerate(teacher_layers):
            s_hidden = student_hiddens[s_layer]
            t_hidden = teacher_hiddens[t_layer]
            
            # Flatten
            s_flat = s_hidden.view(-1, s_hidden.size(-1))
            t_flat = t_hidden.view(-1, t_hidden.size(-1))
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1).mean().item()
            matrix[i, j] = cos_sim
    
    return matrix


if __name__ == "__main__":
    # Example usage
    tracker = MetricsTracker()
    
    # Simulate some predictions
    tracker.add_batch(
        predictions=["42", "17", "100"],
        ground_truths=["42", "18", "95"],
        confidences=[0.95, 0.72, 0.88],
        layer_similarities={8: 0.85, 9: 0.87, 10: 0.89},
        difficulties=["easy", "medium", "hard"]
    )
    
    # Print summary
    tracker.print_summary()
