"""
Code evaluation utilities and evaluator class.
"""

import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

try:
    from tqdm import tqdm
    import datasets
    from vllm import SamplingParams
except ImportError:
    # Handle missing dependencies gracefully
    pass

from rewards.code import compute_score
from .base_evaluator import BaseEvaluator


class CodeEvaluator(BaseEvaluator):
    """Evaluator for code generation tasks."""
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset based on name."""
        if dataset_path == 'lcb':
            return self._load_livecodebench()
        elif dataset_path == 'mbpp':
            return self._load_mbpp()
        elif dataset_path == 'he':
            return self._load_humaneval()
        else:
            raise ValueError(f"Unknown dataset: {dataset_path}")
    
    def _load_livecodebench(self) -> List[Dict[str, Any]]:
        """Load LiveCodeBench dataset."""
        import zlib
        import base64
        
        def process_example(example: Dict) -> Dict[str, Any]:
            prompt = (
                "You will be given a question (problem specification) and will generate "
                "a correct Python program that matches the specification and passes all tests.\n\n"
                f"Question:\n{example['question_content']}"
            )
            
            if example['starter_code']:
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
            
            # Process test cases
            public_tests = json.loads(example.pop('public_test_cases'))
            private_tests = example.pop('private_test_cases')
            
            try:
                private_tests = json.loads(private_tests)
            except json.JSONDecodeError:
                # Handle compressed test cases
                private_tests = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(private_tests.encode("utf-8"))
                        )
                    )
                )
            
            all_tests = public_tests + private_tests
            eval_types = [
                "call" if test['testtype'] == "functional" else "stdio" 
                for test in all_tests
            ]
            
            # Ensure consistent evaluation type
            if not all(et == eval_types[0] for et in eval_types):
                raise ValueError("Mixed evaluation types not supported")
            
            metadata = json.loads(example['metadata'])
            
            return {
                "prompt": prompt,
                "instruction": instruction,
                "ground_truth": {
                    'eval_type': eval_types[0],
                    "fn_name": metadata.get("func_name"),
                    'input_output': {
                        "inputs": [test['input'] for test in all_tests],
                        "outputs": [test['output'] for test in all_tests]
                    }
                }
            }
        
        dataset = datasets.load_dataset(
            "livecodebench/code_generation_lite", 
            version_tag="release_v6", 
            trust_remote_code=True
        )["test"].sort("question_id")
        
        return [process_example(example) for example in tqdm(dataset, desc="Loading LiveCodeBench")]
    
    def _load_mbpp(self, split: str = 'test') -> List[Dict[str, Any]]:
        """Load MBPP dataset."""
        def process_example(example: Dict) -> Dict[str, Any]:
            illustrative_tests = '\n'.join(example['test_list'][:3])
            all_test_cases = example['test_list'] + example['challenge_test_list']
            test_cases = [
                example['test_setup_code'] + "\n" + case 
                for case in all_test_cases
            ]
            
            instruction = (
                "Reason about the problem and any base cases before writing the code. "
                "You must return the implementation code in the following format:\n"
                "```python\n<CODE GOES HERE>\n```\n\n"
            )
            
            prompt = (
                f"{example['text']}\n\n"
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
        
        dataset = datasets.load_dataset("nlile/mbpp")[split]
        return [process_example(example) for example in tqdm(dataset, desc="Loading MBPP")]
    
    def _load_humaneval(self) -> List[Dict[str, Any]]:
        """Load HumanEval dataset."""
        def process_example(example: Dict) -> Dict[str, Any]:
            prompt = (
                "You will be given a code outline and will generate a correct Python program "
                "that matches the specification and passes all tests.\n\n"
                f"```python\n{example['prompt']}\n```\n\n"
            )
            
            instruction = (
                "Reason about the problem and any base cases before writing the code. "
                "You must return the implementation code in the following format:\n"
                "```python\n<CODE GOES HERE>\n```\n\n"
            )
            
            test_input = f"{example['test']}\n\ncheck({example['entry_point']})"
            
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
        
        dataset = datasets.load_dataset("openai/openai_humaneval")['test']
        return [process_example(example) for example in tqdm(dataset, desc="Loading HumanEval")]
    
    def build_aggregate_prompt(
        self, 
        question: str, 
        candidates: List[str], 
        instruction: str
    ) -> str:
        """Build aggregation prompt for code generation."""
        parts = []
        
        if len(candidates) == 1:
            parts.append(
                "You are given a python code implementation problem and a candidate code block "
                "with its reasoning. The candidate may be incomplete or contain errors. "
                "Refine this trajectory and produce an improved, higher-quality solution. "
                "If it is entirely wrong, attempt a new strategy."
            )
        else:
            parts.append(
                "You are given a python code implementation problem and several candidate code blocks "
                "with their reasoning. Some candidates may be incorrect or contain errors. "
                "Aggregate the useful ideas and produce a single, high-quality solution. "
                "Reason carefully; if candidates disagree, choose the correct path."
            )
        
        parts.append(question.strip() + "\n")
        
        if len(candidates) == 1:
            parts.append("Candidate solution (may contain mistakes):\n")
            parts.append(f"---- Candidate ----\n{candidates[0].strip()}\n")
        else:
            parts.append("Candidate solutions (may contain mistakes):\n")
            for i, candidate in enumerate(candidates, 1):
                parts.append(f"---- Solution {i} ----\n{candidate.strip()}\n")
        
        parts.append("\nNow provide an improved and correct solution along with its reasoning.")
        parts.append(instruction)
        
        return "\n".join(parts)
    
    def build_verification_prompt(self, question: str, candidate: str) -> str:
        """Build verification prompt for code solutions."""
        return (
            "You are given a problem and a candidate solution. "
            "Verify whether the candidate solution is correct. "
            "If the solution is correct, output only True. "
            "If it is incorrect, output only False. "
            "Do not generate anything else.\n\n"
            f"Problem:\n{question.strip()}\n\n"
            f"Candidate solution:\n{candidate.strip()}\n\n"
            "Now verify if the solution is True or False. Only output \"True\" or \"False\"."
        )
    
    def evaluate_responses(
        self, 
        responses: List[str], 
        ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate code responses against ground truth."""
        # Use ProcessPoolExecutor for parallel evaluation
        with ProcessPoolExecutor(max_workers=48) as executor:
            futures = [
                executor.submit(compute_score, response, ground_truth, False)
                for response in responses
            ]
            correct_results = [future.result() for future in futures]
        
        mean_acc = sum(correct_results) / max(1, len(correct_results))
        pass_at_k = 1.0 if any(correct_results) else 0.0
        
        return {
            "mean_acc": mean_acc,
            "pass_at_k": pass_at_k,
        }
    
    def verify_responses(
        self, 
        questions: List[str], 
        responses: List[str], 
        temperature: float = 0.1
    ) -> List[int]:
        """Verify multiple responses using the model."""
        verification_prompts = [
            self.render_chat_template(self.build_verification_prompt(q, r))
            for q, r in zip(questions, responses)
        ]
        
        sampling_params = SamplingParams(
            n=1, 
            temperature=temperature, 
            max_tokens=10
        )
        
        outputs = self.llm.generate(verification_prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        # Parse True/False responses
        verified = [
            1 if re.search(r'\btrue\b', response, re.IGNORECASE) else 0
            for response in responses
        ]
        
        return verified