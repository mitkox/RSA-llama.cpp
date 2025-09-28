"""
Base evaluator class with common functionality for both code and math/reasoning evaluation.
"""

import json
import os
import pickle
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class BaseEvaluator(ABC):
    """Base class for evaluation tasks with common functionality."""
    
    def __init__(
        self,
        model_name: str,
        tp_size: int = 4,
        dtype: str = "bfloat16",
        seed: int = 1234,
    ):
        self.model_name = model_name
        self.tp_size = tp_size
        self.dtype = dtype
        self.seed = seed
        
        # Initialize tokenizer and model
        self.tokenizer = self._initialize_tokenizer()
        self.llm = self._initialize_llm()
        
        # Set random seeds
        self._set_seeds()
    
    def _initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize the tokenizer."""
        return AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
    
    def _initialize_llm(self) -> LLM:
        """Initialize the LLM with appropriate configuration."""
        kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tp_size,
            "dtype": self.dtype,
            "trust_remote_code": True,
            "seed": self.seed,
        }
        
        # Special handling for nemo models
        if 'nemo' in self.model_name.lower():
            kwargs["mamba_ssm_cache_dtype"] = 'float32'
        
        return LLM(**kwargs)
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    @abstractmethod
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load and process the dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def evaluate_responses(
        self, 
        responses: List[str], 
        ground_truth: Any
    ) -> Dict[str, float]:
        """Evaluate responses against ground truth. Must be implemented by subclasses."""
        pass
    
    def render_chat_template(self, prompt: str) -> str:
        """Render a prompt using the chat template."""
        chat_message = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            chat_message, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_candidates(
        self, 
        candidate_pool: Optional[List[str]], 
        population: int, 
        k: int
    ) -> List[Optional[List[str]]]:
        """Generate candidate combinations for aggregation."""
        if candidate_pool is None:
            return [None] * population
        
        if len(candidate_pool) < k:
            # If not enough candidates, pad with None or repeat
            return [candidate_pool] * population
        
        return [random.sample(candidate_pool, k) for _ in range(population)]
    
    def reshape_responses(self, responses: List[str], population: int) -> List[List[str]]:
        """Reshape flat response list into groups of population size."""
        return [responses[i:i+population] for i in range(0, len(responses), population)]
    
    def calculate_response_statistics(self, responses: List[str]) -> Dict[str, float]:
        """Calculate statistics about response lengths."""
        lengths = [len(self.tokenizer.encode(response)) for response in responses]
        
        return {
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.percentile(lengths, 50)),
            "q25_length": float(np.percentile(lengths, 25)),
            "q75_length": float(np.percentile(lengths, 75)),
        }
    
    def save_checkpoint(
        self, 
        data: List[Dict[str, Any]], 
        checkpoint_path: str, 
        loop_idx: int
    ) -> None:
        """Save evaluation checkpoint."""
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f'loop_{loop_idx}.pkl')
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load_latest_checkpoint(self, checkpoint_dir: str) -> tuple[List[Dict], int, Optional[Path]]:
        """Load the most recent checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        pattern = re.compile(r"loop_(\d+)\.pkl$")
        
        max_idx = -1
        latest_file = None
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist")
        
        for file in checkpoint_path.iterdir():
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                        latest_file = file
        
        if latest_file is None:
            raise FileNotFoundError("No checkpoint files found in directory")
        
        with open(latest_file, "rb") as f:
            data = pickle.load(f)
        
        return data, max_idx, latest_file
    
    def append_metrics_to_json(self, path: str, entry: Dict[str, Any]) -> None:
        """Append metrics entry to JSON file."""
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            else:
                data = []
        except (json.JSONDecodeError, IOError):
            # Handle corrupted or unreadable files
            data = []
        
        data.append(entry)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)