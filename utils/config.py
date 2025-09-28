"""
Utilities and configuration management for RSA evaluation.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs with validation."""
    
    # Model configuration
    model_name: str
    tp_size: int = 4
    dtype: str = "bfloat16"
    
    # Generation parameters
    max_new_tokens: int = 8192
    temperature: float = 1.0
    
    # Evaluation parameters
    loops: int = 2
    k: int = 4  # Number of candidates to sample
    population: int = 4  # Population size per iteration
    
    # System parameters
    seed: int = 1234
    resume: bool = False
    self_verify: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        if self.loops < 1:
            errors.append("loops must be >= 1")
        if self.k < 1:
            errors.append("k must be >= 1")
        if self.population < 1:
            errors.append("population must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            errors.append("temperature must be between 0.0 and 2.0")
        if self.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1")
        if self.tp_size < 1:
            errors.append("tp_size must be >= 1")
        
        valid_dtypes = {"auto", "float16", "bfloat16"}
        if self.dtype not in valid_dtypes:
            errors.append(f"dtype must be one of {valid_dtypes}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def setup_output_directory(base_path: str, model_name: str, config: EvaluationConfig) -> Dict[str, str]:
    """Setup output directory structure and return paths."""
    model_dir = model_name.split('/')[-1]
    output_dir = os.path.join(base_path, model_dir)
    
    config_str = f"k_{config.k}_N_{config.population}_seed_{config.seed}"
    metrics_path = os.path.join(output_dir, f'{config_str}.json')
    checkpoints_path = os.path.join(output_dir, 'checkpoints', config_str)
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    
    return {
        'output_dir': output_dir,
        'metrics_path': metrics_path,
        'checkpoints_path': checkpoints_path
    }


def validate_dataset_choice(dataset: str) -> str:
    """Validate dataset choice and return normalized name."""
    valid_datasets = {'lcb', 'mbpp', 'he', 'humaneval'}
    
    # Normalize dataset name
    dataset_normalized = dataset.lower()
    if dataset_normalized == 'humaneval':
        dataset_normalized = 'he'
    
    if dataset_normalized not in valid_datasets:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid choices: {valid_datasets}")
    
    return dataset_normalized


def print_evaluation_summary(config: EvaluationConfig, dataset: str, data_size: int) -> None:
    """Print a summary of the evaluation configuration."""
    print("=" * 60)
    print("RSA EVALUATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {dataset} ({data_size} examples)")
    print(f"Loops: {config.loops}")
    print(f"K (candidates): {config.k}")
    print(f"Population: {config.population}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_new_tokens}")
    print(f"Seed: {config.seed}")
    print(f"Resume: {config.resume}")
    print(f"Self-verify: {config.self_verify}")
    print("=" * 60)


# Constants for better maintainability
SUPPORTED_DATASETS = {
    'lcb': 'LiveCodeBench',
    'mbpp': 'Mostly Basic Python Problems',
    'he': 'HumanEval'
}

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = "eval/"

# Error messages
ERROR_MESSAGES = {
    'model_init': "Failed to initialize model. Check model name and CUDA availability.",
    'dataset_load': "Failed to load dataset. Check internet connection and dataset availability.",
    'checkpoint_load': "Failed to load checkpoint. File may be corrupted.",
    'metrics_save': "Failed to save metrics. Check disk space and permissions.",
}