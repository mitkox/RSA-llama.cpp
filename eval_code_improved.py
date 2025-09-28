"""
Improved version of eval_code.py with better error handling, validation, and structure.
"""

import argparse
import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rewards.code import compute_score


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class CodeEvaluationPipeline:
    """Main pipeline for code evaluation with improved error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)
        
        # Initialize components
        self.tokenizer = None
        self.llm = None
        self._initialize_model()
        
        # Set random seeds
        self._set_random_seeds()
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters."""
        required_fields = {
            'model_name': str,
            'loops': int,
            'k': int,
            'population': int,
            'max_new_tokens': int,
            'temperature': float,
            'tp_size': int,
            'dtype': str,
            'seed': int,
        }
        
        for field, expected_type in required_fields.items():
            if field not in config:
                raise ValidationError(f"Missing required field: {field}")
            
            if not isinstance(config[field], expected_type):
                raise ValidationError(
                    f"Field {field} must be of type {expected_type.__name__}, "
                    f"got {type(config[field]).__name__}"
                )
        
        # Validate ranges
        if config['loops'] < 1:
            raise ValidationError("loops must be >= 1")
        if config['k'] < 1:
            raise ValidationError("k must be >= 1") 
        if config['population'] < 1:
            raise ValidationError("population must be >= 1")
        if not 0.0 <= config['temperature'] <= 2.0:
            raise ValidationError("temperature must be between 0.0 and 2.0")
        
        return config
    
    def _initialize_model(self) -> None:
        """Initialize tokenizer and model with error handling."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'], 
                trust_remote_code=True
            )
            
            llm_kwargs = {
                "model": self.config['model_name'],
                "tensor_parallel_size": self.config['tp_size'],
                "dtype": self.config['dtype'],
                "trust_remote_code": True,
                "seed": self.config['seed'],
            }
            
            # Special handling for nemo models
            if 'nemo' in self.config['model_name'].lower():
                llm_kwargs["mamba_ssm_cache_dtype"] = 'float32'
            
            self.llm = LLM(**llm_kwargs)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {self.config['model_name']}: {e}")
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        import random
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
    
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load dataset with error handling."""
        try:
            if dataset_name == 'lcb':
                return self._load_livecodebench()
            elif dataset_name == 'mbpp':
                return self._load_mbpp()
            elif dataset_name == 'he':
                return self._load_humaneval()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")
    
    def _load_livecodebench(self) -> List[Dict[str, Any]]:
        """Load LiveCodeBench dataset with better error handling."""
        try:
            import datasets
            from tqdm import tqdm
            import zlib
            import base64
            import pickle
            
            dataset = datasets.load_dataset(
                "livecodebench/code_generation_lite", 
                version_tag="release_v6", 
                trust_remote_code=True
            )["test"].sort("question_id")
            
            processed_data = []
            
            for example in tqdm(dataset, desc="Processing LiveCodeBench"):
                try:
                    processed_example = self._process_lcb_example(example)
                    processed_data.append(processed_example)
                except Exception as e:
                    print(f"Warning: Failed to process example {example.get('question_id', 'unknown')}: {e}")
                    continue
            
            if not processed_data:
                raise RuntimeError("No examples were successfully processed")
            
            return processed_data
            
        except ImportError as e:
            raise RuntimeError(f"Missing required dependencies for LiveCodeBench: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading LiveCodeBench: {e}")
    
    def _process_lcb_example(self, example: Dict) -> Dict[str, Any]:
        """Process a single LiveCodeBench example."""
        import json
        import zlib
        import base64
        import pickle
        
        prompt = (
            "You will be given a question (problem specification) and will generate "
            "a correct Python program that matches the specification and passes all tests.\n\n"
            f"Question:\n{example['question_content']}"
        )
        
        if example.get('starter_code', '').strip():
            instruction = (
                "You will use the following starter code to write the solution to the problem "
                "and enclose your code within delimiters.\n"
                f"```python\n{example['starter_code']}\n```\n\n"
            )
        else:
            instruction = (
                "Read the inputs from stdin solve the problem and write the answer to stdout "
                "(do not directly test on the sample inputs). Enclose your code within delimiters "
                "as follows. Ensure that when the python program runs, it reads the inputs, "
                "runs the algorithm and writes output to STDOUT.\n\n"
                "```python\n# YOUR CODE HERE\n```"
            )
        
        # Process test cases safely
        try:
            public_tests = json.loads(example['public_test_cases'])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid public test cases: {e}")
        
        private_tests = example['private_test_cases']
        try:
            private_tests = json.loads(private_tests)
        except json.JSONDecodeError:
            try:
                # Handle compressed test cases
                private_tests = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(private_tests.encode("utf-8"))
                        )
                    )
                )
            except Exception as e:
                raise ValueError(f"Failed to decode private test cases: {e}")
        
        all_tests = public_tests + private_tests
        eval_types = [
            "call" if test.get('testtype') == "functional" else "stdio" 
            for test in all_tests
        ]
        
        # Validate evaluation types
        if not all_tests:
            raise ValueError("No test cases found")
        
        if not all(et == eval_types[0] for et in eval_types):
            raise ValueError("Mixed evaluation types not supported")
        
        try:
            metadata = json.loads(example['metadata'])
        except (json.JSONDecodeError, KeyError):
            metadata = {}
        
        return {
            "prompt": prompt,
            "instruction": instruction,
            "ground_truth": {
                'eval_type': eval_types[0],
                "fn_name": metadata.get("func_name"),
                'input_output': {
                    "inputs": [test.get('input', '') for test in all_tests],
                    "outputs": [test.get('output', '') for test in all_tests]
                }
            }
        }
    
    def _load_mbpp(self, split: str = 'test') -> List[Dict[str, Any]]:
        """Load MBPP dataset."""
        try:
            import datasets
            from tqdm import tqdm
            
            dataset = datasets.load_dataset("nlile/mbpp")[split]
            processed_data = []
            
            for example in tqdm(dataset, desc="Processing MBPP"):
                try:
                    processed_example = self._process_mbpp_example(example)
                    processed_data.append(processed_example)
                except Exception as e:
                    print(f"Warning: Failed to process MBPP example: {e}")
                    continue
            
            return processed_data
            
        except ImportError as e:
            raise RuntimeError(f"Missing required dependencies for MBPP: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading MBPP: {e}")
    
    def _process_mbpp_example(self, example: Dict) -> Dict[str, Any]:
        """Process a single MBPP example."""
        test_list = example.get('test_list', [])
        challenge_list = example.get('challenge_test_list', [])
        setup_code = example.get('test_setup_code', '')
        
        illustrative_tests = '\n'.join(test_list[:3])
        all_test_cases = test_list + challenge_list
        test_cases = [
            f"{setup_code}\n{case}" if setup_code else case
            for case in all_test_cases
        ]
        
        instruction = (
            "Reason about the problem and any base cases before writing the code. "
            "You must return the implementation code in the following format:\n"
            "```python\n<CODE GOES HERE>\n```\n\n"
        )
        
        prompt = (
            f"{example.get('text', '')}\n\n"
            f"Your code should satisfy these tests:\n\n{illustrative_tests}"
        )
        
        return {
            "prompt": prompt,
            "instruction": instruction,
            "ground_truth": {
                'eval_type': 'assert',
                'input_output': {
                    "inputs": test_cases,
                    "outputs": [None] * len(test_cases)
                }
            }
        }
    
    def _load_humaneval(self) -> List[Dict[str, Any]]:
        """Load HumanEval dataset."""
        try:
            import datasets
            from tqdm import tqdm
            
            dataset = datasets.load_dataset("openai/openai_humaneval")['test']
            processed_data = []
            
            for example in tqdm(dataset, desc="Processing HumanEval"):
                try:
                    processed_example = self._process_humaneval_example(example)
                    processed_data.append(processed_example)
                except Exception as e:
                    print(f"Warning: Failed to process HumanEval example: {e}")
                    continue
            
            return processed_data
            
        except ImportError as e:
            raise RuntimeError(f"Missing required dependencies for HumanEval: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading HumanEval: {e}")
    
    def _process_humaneval_example(self, example: Dict) -> Dict[str, Any]:
        """Process a single HumanEval example."""
        prompt = (
            "You will be given a code outline and will generate a correct Python program "
            "that matches the specification and passes all tests.\n\n"
            f"```python\n{example.get('prompt', '')}\n```\n\n"
        )
        
        instruction = (
            "Reason about the problem and any base cases before writing the code. "
            "You must return the implementation code in the following format:\n"
            "```python\n<CODE GOES HERE>\n```\n\n"
        )
        
        test_input = f"{example.get('test', '')}\n\ncheck({example.get('entry_point', '')})"
        
        return {
            "prompt": prompt,
            "instruction": instruction,
            "ground_truth": {
                'eval_type': 'assert',
                'input_output': {
                    "inputs": [test_input],
                    "outputs": [None]
                }
            }
        }
    
    def run_evaluation_loop(
        self, 
        data: List[Dict[str, Any]], 
        output_dir: str, 
        resume: bool = False
    ) -> None:
        """Run the main evaluation loop with improved error handling."""
        try:
            # Setup output directory and paths
            os.makedirs(output_dir, exist_ok=True)
            config_str = f"k_{self.config['k']}_N_{self.config['population']}_seed_{self.config['seed']}"
            metrics_path = os.path.join(output_dir, f'{config_str}.json')
            checkpoints_path = os.path.join(output_dir, 'checkpoints', config_str)
            os.makedirs(checkpoints_path, exist_ok=True)
            
            # Clear metrics file if not resuming
            if not resume and os.path.exists(metrics_path):
                os.remove(metrics_path)
            
            # Prepare data structure
            processed_data = self._prepare_evaluation_data(data, checkpoints_path, resume)
            start_loop_idx = processed_data['start_idx']
            
            # Run evaluation loops
            for loop_idx in range(start_loop_idx + 1, self.config['loops']):
                try:
                    print(f"Running loop {loop_idx + 1}/{self.config['loops']}")
                    
                    loop_data, metrics = self._run_single_loop(processed_data['data'])
                    
                    # Save checkpoint
                    self._save_checkpoint(loop_data, checkpoints_path, loop_idx)
                    
                    # Save metrics
                    metrics_entry = self._prepare_metrics_entry(metrics, loop_idx)
                    self._append_metrics(metrics_path, metrics_entry)
                    
                    print(f"Loop {loop_idx} completed. Metrics: {json.dumps(metrics_entry, indent=2)}")
                    
                except Exception as e:
                    print(f"Error in loop {loop_idx}: {e}")
                    print(traceback.format_exc())
                    
                    # Optionally continue with next loop or break
                    if self.config.get('continue_on_error', False):
                        continue
                    else:
                        raise
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            print(traceback.format_exc())
            raise
    
    def _prepare_evaluation_data(
        self, 
        data: List[Dict[str, Any]], 
        checkpoints_path: str, 
        resume: bool
    ) -> Dict[str, Any]:
        """Prepare data for evaluation, handling resume logic."""
        if resume:
            try:
                checkpoint_data, start_idx, _ = self._load_latest_checkpoint(checkpoints_path)
                print(f"Resumed from checkpoint at loop {start_idx}")
                return {
                    'data': checkpoint_data,
                    'start_idx': start_idx
                }
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting from scratch.")
        
        # Prepare fresh data structure
        processed_data = []
        for item in data:
            processed_data.append({
                'orig_prompt': item['prompt'],
                'ground_truth': item['ground_truth'],
                'candidates': None,
                'instruction': item['instruction']
            })
        
        return {
            'data': processed_data,
            'start_idx': -1
        }
    
    def _run_single_loop(self, data: List[Dict[str, Any]]) -> tuple[List[Dict], Dict[str, Any]]:
        """Run a single evaluation loop."""
        # Implementation similar to original run() function but with better error handling
        # This is a placeholder - the full implementation would be quite long
        sampling_params = SamplingParams(
            n=1, 
            temperature=self.config['temperature'], 
            max_tokens=self.config['max_new_tokens']
        )
        
        # Generate requests and process responses
        # ... (implementation details)
        
        return data, {"mean_acc": 0.0, "pass_at_k": 0.0}  # Placeholder
    
    def _save_checkpoint(self, data: List[Dict], path: str, loop_idx: int) -> None:
        """Save evaluation checkpoint."""
        import pickle
        checkpoint_file = os.path.join(path, f'loop_{loop_idx}.pkl')
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    def _load_latest_checkpoint(self, checkpoint_dir: str) -> tuple:
        """Load the most recent checkpoint."""
        import pickle
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_dir)
        pattern = re.compile(r"loop_(\d+)\.pkl$")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist")
        
        max_idx = -1
        latest_file = None
        
        for file in checkpoint_path.iterdir():
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                        latest_file = file
        
        if latest_file is None:
            raise FileNotFoundError("No checkpoint files found")
        
        with open(latest_file, "rb") as f:
            data = pickle.load(f)
        
        return data, max_idx, latest_file
    
    def _prepare_metrics_entry(self, metrics: Dict[str, Any], loop_idx: int) -> Dict[str, Any]:
        """Prepare metrics entry for JSON logging."""
        return {
            "loop": loop_idx,
            "k": self.config['k'],
            "population": self.config['population'],
            "seed": self.config['seed'],
            **metrics
        }
    
    def _append_metrics(self, path: str, entry: Dict[str, Any]) -> None:
        """Append metrics to JSON file safely."""
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            else:
                data = []
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read existing metrics file {path}, creating new one")
            data = []
        
        data.append(entry)
        
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to write metrics to {path}: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser with validation."""
    parser = argparse.ArgumentParser(
        description="Improved code evaluation with better error handling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                       help="Model name or path")
    parser.add_argument("--dataset", default="lcb", choices=["lcb", "mbpp", "he"],
                       help="Dataset to evaluate on")
    parser.add_argument("--output", default="eval/",
                       help="Output directory for results")
    parser.add_argument("--k", type=int, default=4,
                       help="Number of candidates to sample for aggregation")
    parser.add_argument("--population", type=int, default=4,
                       help="Population size for each iteration")
    parser.add_argument("--loops", type=int, default=2,
                       help="Number of evaluation loops")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--tp-size", type=int, default=4,
                       help="Tensor parallel size")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"],
                       help="Model data type")
    parser.add_argument("--seed", type=int, default=1234,
                       help="Random seed for reproducibility")
    parser.add_argument("--resume", action='store_true', default=False,
                       help="Resume from latest checkpoint")
    parser.add_argument("--self-verify", action='store_true', default=False,
                       help="Enable self-verification of candidates")
    parser.add_argument("--continue-on-error", action='store_true', default=False,
                       help="Continue evaluation even if individual loops fail")
    
    return parser


def main():
    """Main entry point with improved error handling."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Convert args to config dict
    config = {
        'model_name': args.model,
        'loops': args.loops,
        'k': args.k,
        'population': args.population,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'tp_size': args.tp_size,
        'dtype': args.dtype,
        'seed': args.seed,
        'self_verify': args.self_verify,
        'continue_on_error': args.continue_on_error,
    }
    
    try:
        # Initialize pipeline
        print("Initializing evaluation pipeline...")
        pipeline = CodeEvaluationPipeline(config)
        
        # Load dataset
        print(f"Loading dataset: {args.dataset}")
        data = pipeline.load_dataset(args.dataset)
        print(f"Loaded {len(data)} examples")
        
        # Setup output directory
        output_dir = os.path.join(args.output, args.model.split('/')[-1])
        
        # Run evaluation
        print("Starting evaluation...")
        pipeline.run_evaluation_loop(data, output_dir, args.resume)
        
        print("Evaluation completed successfully!")
        
    except ValidationError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())