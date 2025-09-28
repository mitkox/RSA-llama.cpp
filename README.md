# RSA Evaluation with llama.cpp Server

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

**Recursive Self-Augmentation (RSA)** evaluation framework adapted to work with [llama.cpp](https://github.com/ggerganov/llama.cpp) server, enabling cross-platform code generation model evaluation without GPU memory limitations.

## 🚀 Key Features

- **Cross-Platform**: Works on Windows, macOS, and Linux
- **No GPU Memory Limits**: Uses HTTP API instead of loading models locally
- **Flexible Deployment**: Model server can run on different machine
- **Multiple Datasets**: Support for LiveCodeBench, MBPP, and HumanEval
- **Iterative Improvement**: Self-augmentation through candidate aggregation

## 📋 Prerequisites

- Python 3.8+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) server
- GGUF format model (e.g., CodeQwen, Code Llama, DeepSeek-Coder)

## 🛠️ Installation

### 1. Install Python Dependencies

```bash
pip install numpy requests openai datasets tqdm
```

### 2. Setup llama.cpp Server

```bash
# Clone and compile llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# For GPU acceleration (optional)
make LLAMA_CUDA=1  # NVIDIA GPUs
make LLAMA_METAL=1 # Apple Silicon
```

### 3. Download a Model

```bash
# Example: CodeQwen1.5-7B-Chat
huggingface-cli download Qwen/CodeQwen1.5-7B-Chat-GGUF codeqwen-1_5-7b-chat-q4_k_m.gguf
```

### 4. Start llama.cpp Server

```bash
./server -m codeqwen-1_5-7b-chat-q4_k_m.gguf --port 8080 --host 0.0.0.0 --n-gpu-layers 32
```

## 🎯 Quick Start

### Test Connection

```bash
python test_llamacpp_connection.py --server-url http://localhost:8080
```

### Run Evaluation

```bash
# LiveCodeBench evaluation
python eval_code.py --dataset lcb --server-url http://localhost:8080

# MBPP evaluation
python eval_code.py --dataset mbpp --server-url http://localhost:8080

# HumanEval evaluation
python eval_code.py --dataset he --server-url http://localhost:8080
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset: `lcb`, `mbpp`, `he` | `lcb` |
| `--server-url` | llama.cpp server URL | `http://localhost:8080` |
| `--loops` | Number of evaluation loops | `2` |
| `--k` | Candidates to sample | `4` |
| `--population` | Population size per iteration | `4` |
| `--temperature` | Sampling temperature | `1.0` |
| `--max-new-tokens` | Maximum tokens to generate | `8192` |
| `--resume` | Resume from checkpoint | `False` |

## 📊 Advanced Usage

### Custom Configuration

```bash
python eval_code.py \
  --dataset lcb \
  --server-url http://localhost:8080 \
  --loops 10 \
  --k 8 \
  --population 16 \
  --temperature 0.7 \
  --max-new-tokens 4096
```

### Remote Server

```bash
python eval_code.py \
  --dataset mbpp \
  --server-url http://192.168.1.100:8080 \
  --server-timeout 600
```

## 🏗️ Architecture

The RSA evaluation framework uses:

1. **Iterative Self-Improvement**: Models generate multiple candidate solutions
2. **Candidate Aggregation**: Best solutions are combined and refined
3. **Self-Verification**: Optional verification of generated solutions
4. **Checkpoint System**: Resume long-running evaluations

## 🔧 Troubleshooting

### Server Connection Issues
```bash
# Check server health
curl http://localhost:8080/health

# Verify server is running
ps aux | grep server
```

### Memory Issues
- Reduce `--ctx-size` when starting llama.cpp
- Use smaller quantized models (Q4_K_M)
- Decrease `--max-new-tokens`

## 📈 Supported Models

Recommended GGUF models for code evaluation:
- **CodeQwen1.5-7B-Chat**: Excellent performance, good speed
- **Code Llama 13B Instruct**: Strong capabilities, larger context
- **DeepSeek-Coder**: Specialized for coding tasks
- **Magicoder**: Fine-tuned for code generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original RSA methodology and implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient inference
- Hugging Face for model hosting and datasets