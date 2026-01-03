"""
Model definitions for Teacher and Student models.
Handles loading pretrained models and extracting hidden states.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Tuple
import logging
import warnings

# Setup logging
logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    """Teacher model with latent reasoning capabilities (COCONUT-style)."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        super().__init__()
        
        # Validate inputs
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if device not in ["cuda", "cpu", "auto"]:
            warnings.warn(f"Unusual device '{device}', expected 'cuda', 'cpu', or 'auto'")
        
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        logger.info(f"Loading teacher model: {model_name}")
        logger.info(f"Device: {device}, 4-bit quantization: {load_in_4bit}")
        
        try:
            # Load model with 4-bit quantization for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get model config
            self.config = self.model.config
            self.num_layers = self.config.num_hidden_layers
            self.hidden_size = self.config.hidden_size
            
            logger.info(f"Model loaded successfully: {self.num_layers} layers, hidden size {self.hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {str(e)}")
            raise RuntimeError(f"Teacher model loading failed: {str(e)}") from e
        
        self.model.eval()  # Teacher is frozen
        logger.info("Teacher model set to eval mode (frozen)")
        
    def _validate_layers(self, layers: Optional[List[int]]) -> List[int]:
        """
        Validate layer indices.
        
        Args:
            layers: List of layer indices to validate
            
        Returns:
            Validated list of layer indices
            
        Raises:
            ValueError: If layer indices are invalid
        """
        if layers is None:
            # Default: last 5 layers
            layers = list(range(max(0, self.num_layers - 5), self.num_layers))
            logger.debug(f"Using default layers: {layers}")
        
        # Validate layer indices
        for layer_idx in layers:
            if not isinstance(layer_idx, int):
                raise TypeError(f"Layer index must be int, got {type(layer_idx)}")
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    f"Layer index {layer_idx} out of range [0, {self.num_layers})"
                )
        
        return layers
        
    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Validate input tensors.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check dimensions
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch, seq], got shape {input_ids.shape}")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D [batch, seq], got shape {attention_mask.shape}")
        
        # Check shape consistency
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}"
            )
        
        # Check batch size
        batch_size, seq_len = input_ids.shape
        if batch_size == 0:
            raise ValueError("Batch size cannot be 0")
        if seq_len == 0:
            raise ValueError("Sequence length cannot be 0")
        
        # Check for valid token IDs
        vocab_size = self.config.vocab_size
        if input_ids.max() >= vocab_size:
            raise ValueError(f"Token ID {input_ids.max()} exceeds vocab size {vocab_size}")
        if input_ids.min() < 0:
            raise ValueError(f"Negative token ID found: {input_ids.min()}")
        
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from specified layers.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            layers: List of layer indices to extract (default: last 5 layers)
            
        Returns:
            Dictionary mapping layer index to hidden states [batch_size, seq_len, hidden_dim]
            
        Raises:
            ValueError: If inputs or layer indices are invalid
        """
        # Validate inputs
        self._validate_inputs(input_ids, attention_mask)
        layers = self._validate_layers(layers)
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise RuntimeError(f"Model forward pass failed: {str(e)}") from e
                
        hidden_states = {}
        for layer_idx in layers:
            try:
                # Apply layer normalization for consistent scale
                hidden = outputs.hidden_states[layer_idx]
                
                # Validate hidden state shape
                expected_shape = (input_ids.shape[0], input_ids.shape[1], self.hidden_size)
                if hidden.shape != expected_shape:
                    logger.warning(
                        f"Layer {layer_idx} shape mismatch: "
                        f"expected {expected_shape}, got {hidden.shape}"
                    )
                
                hidden_states[layer_idx] = nn.functional.layer_norm(
                    hidden,
                    normalized_shape=(hidden.size(-1),)
                )
            except IndexError as e:
                logger.error(f"Layer {layer_idx} not found in model outputs")
                raise IndexError(f"Layer {layer_idx} does not exist") from e
            
        logger.debug(f"Extracted hidden states from {len(hidden_states)} layers")
        return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with hidden state extraction.
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden_states: Dictionary of hidden states from specified layers
        """
        hidden_states = self.extract_hidden_states(input_ids, attention_mask, layers)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
        return outputs.logits, hidden_states


class StudentModel(nn.Module):
    """Student model with LoRA adapters for efficient fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = ["q_proj", "v_proj"]
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store LoRA config
        self.lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules
        }
        
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get hidden states from student model (same interface as teacher).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            layers: List of layer indices to extract
            
        Returns:
            Dictionary mapping layer index to hidden states
        """
        if layers is None:
            layers = list(range(8, 13))  # Default: layers 8-12
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = {}
        for layer_idx in layers:
            # Apply layer normalization
            hidden = outputs.hidden_states[layer_idx]
            hidden_states[layer_idx] = nn.functional.layer_norm(
                hidden,
                normalized_shape=(hidden.size(-1),)
            )
            
        return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with hidden state extraction.
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden_states: Dictionary of hidden states from specified layers
            loss: Task loss if labels provided
        """
        hidden_states = self.get_hidden_states(input_ids, attention_mask, layers)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs.logits, hidden_states, outputs.loss if labels is not None else None
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "trainable": trainable_params,
            "total": total_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }


def print_model_info(model: nn.Module, name: str = "Model"):
    """Print detailed model information."""
    if hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        print(f"\n{'='*70}")
        print(f"{name} Information:")
        print(f"{'='*70}")
        print(f"Total parameters:     {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Trainable %:          {params['trainable_percentage']:.4f}%")
        print(f"{'='*70}\n")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"{name}: {total:,} parameters")
