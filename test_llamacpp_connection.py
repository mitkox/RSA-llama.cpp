#!/usr/bin/env python3
"""
Test script to verify llama.cpp server connection and basic functionality.

Usage:
    python test_llamacpp_connection.py --server-url http://localhost:8080
"""

import argparse
import json
import requests
import time
from openai import OpenAI


def test_server_health(server_url: str) -> bool:
    """Test if llama.cpp server is responding."""
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server health check passed")
            return True
        else:
            print(f"❌ Server health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {server_url}: {e}")
        return False


def test_basic_generation(server_url: str, model_name: str = "local-model") -> bool:
    """Test basic text generation."""
    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="local-key"
        )
        
        print("🔄 Testing basic generation...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Write a simple Python function to add two numbers."}],
            max_tokens=200,
            temperature=0.7
        )
        
        generated_text = response.choices[0].message.content
        print("✅ Basic generation test passed")
        print(f"Generated text preview: {generated_text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Basic generation test failed: {e}")
        return False


def test_code_generation(server_url: str, model_name: str = "local-model") -> bool:
    """Test code-specific generation."""
    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="local-key"
        )
        
        print("🔄 Testing code generation...")
        
        test_prompt = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question:
Write a function that takes two integers and returns their sum.

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

```python
# YOUR CODE HERE
```"""
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        generated_code = response.choices[0].message.content
        print("✅ Code generation test passed")
        
        # Check if response contains python code block
        if "```python" in generated_code:
            print("✅ Response contains Python code block")
        else:
            print("⚠️ Response may not contain properly formatted Python code")
        
        print(f"Generated code preview: {generated_code[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Code generation test failed: {e}")
        return False


def test_server_info(server_url: str) -> bool:
    """Get server information."""
    try:
        # Try to get model info
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models_info = response.json()
            print("✅ Server models info retrieved")
            print(f"Available models: {json.dumps(models_info, indent=2)}")
        
        # Try to get server props (if available)
        try:
            response = requests.get(f"{server_url}/props", timeout=10)
            if response.status_code == 200:
                props = response.json()
                print("✅ Server properties retrieved")
                print(f"Server properties: {json.dumps(props, indent=2)}")
        except:
            print("ℹ️ Server properties not available (normal for some llama.cpp versions)")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Could not retrieve server info: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test llama.cpp server connection")
    parser.add_argument("--server-url", default="http://localhost:8080", 
                       help="URL for llama.cpp server")
    parser.add_argument("--model-name", default="local-model", 
                       help="Model name identifier")
    args = parser.parse_args()
    
    print("🚀 Testing llama.cpp server connection...")
    print(f"Server URL: {args.server_url}")
    print(f"Model name: {args.model_name}")
    print("-" * 50)
    
    all_tests_passed = True
    
    # Test 1: Server Health
    if not test_server_health(args.server_url):
        all_tests_passed = False
        print("\n❌ Server is not responding. Make sure llama.cpp server is running:")
        print(f"   ./server -m model.gguf --port 8080 --host 0.0.0.0")
        return 1
    
    # Test 2: Server Info
    test_server_info(args.server_url)
    
    # Test 3: Basic Generation
    if not test_basic_generation(args.server_url, args.model_name):
        all_tests_passed = False
    
    # Test 4: Code Generation
    if not test_code_generation(args.server_url, args.model_name):
        all_tests_passed = False
    
    print("-" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your setup is ready for RSA evaluation.")
        print("\nYou can now run:")
        print(f"   python eval_code.py --server-url {args.server_url} --dataset lcb")
    else:
        print("❌ Some tests failed. Please check your setup.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())