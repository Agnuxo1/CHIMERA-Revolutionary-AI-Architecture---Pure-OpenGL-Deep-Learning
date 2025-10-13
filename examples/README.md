# 🚀 CHIMERA Examples and Demos

This directory contains examples and demonstrations of CHIMERA v3.0 functionality.

## 📋 Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Advanced Examples](#advanced-examples)
- [Interactive Demos](#interactive-demos)
- [Performance Benchmarks](#performance-benchmarks)
- [Research Examples](#research-examples)

---

## 🎯 Quick Start Examples

### 1. Basic Mathematical Operations (`math_operations.py`)

Demonstrates basic OpenGL operations without deep learning.

```bash
python examples/math_operations.py
```

**What it shows:**
- Matrix multiplication on GPU
- Vector operations
- Performance comparison vs CPU

### 2. Self-Attention Demo (`attention_demo.py`)

Shows attention mechanism working on pure OpenGL.

```bash
python examples/attention_demo.py
```

**What it shows:**
- Q, K, V processing
- Attention score computation
- Visualization of attention patterns

### 3. Transformer Block Demo (`transformer_demo.py`)

Complete transformer block running on OpenGL.

```bash
python examples/transformer_demo.py
```

**What it shows:**
- Embedding layer
- Multi-head attention
- Feed-forward network
- Layer normalization

---

## 🔬 Advanced Examples

### 1. Model Conversion (`qwen_conversion.py`)

Convert PyTorch models to run on OpenGL.

```bash
python examples/qwen_conversion.py
```

**What it shows:**
- Loading Qwen model from Hugging Face
- Converting weights to OpenGL textures
- Verification of conversion accuracy

### 2. Custom Training (`custom_training.py`)

Train a model entirely on OpenGL (no PyTorch).

```bash
python examples/custom_training.py
```

**What it shows:**
- Dataset preparation
- Training loop in OpenGL shaders
- Model evaluation and saving

### 3. Multi-GPU Inference (`multi_gpu_demo.py`)

Use multiple GPUs for inference.

```bash
python examples/multi_gpu_demo.py
```

**What it shows:**
- GPU detection and selection
- Load balancing across GPUs
- Performance scaling

---

## 🎮 Interactive Demos

### 1. Chat Interface (`interactive_chat.py`)

Interactive chat with CHIMERA model.

```bash
python examples/interactive_chat.py
```

**Features:**
- Real-time conversation
- Context management
- Response time tracking
- Model switching

### 2. Real-time Generation (`realtime_demo.py`)

See text generation in real-time.

```bash
python examples/realtime_demo.py
```

**Features:**
- Live token generation
- Speed monitoring
- Memory usage tracking
- Interactive parameter adjustment

### 3. Performance Benchmarking (`benchmark_suite.py`)

Comprehensive performance testing.

```bash
python examples/benchmark_suite.py
```

**What it tests:**
- Matrix operations speed
- Memory bandwidth
- GPU utilization
- Cross-platform comparison

---

## 📊 Performance Benchmarks

### 1. Speed Tests (`speed_benchmarks.py`)

Compare CHIMERA vs traditional frameworks.

```bash
python examples/speed_benchmarks.py
```

**Benchmarks:**
- Matrix multiplication (2048×2048)
- Self-attention computation
- Full transformer inference
- Memory allocation/deallocation

### 2. Memory Tests (`memory_benchmarks.py`)

Memory usage and efficiency tests.

```bash
python examples/memory_benchmarks.py
```

**Tests:**
- Peak memory usage
- Memory fragmentation
- GPU memory management
- Cross-batch efficiency

### 3. Scalability Tests (`scalability_benchmarks.py`)

Test performance at different scales.

```bash
python examples/scalability_benchmarks.py
```

**Tests:**
- Different model sizes
- Various batch sizes
- Multi-GPU scaling
- Long sequence handling

---

## 🔬 Research Examples

### 1. Holographic Memory (`holographic_demo.py`)

Demonstrate holographic memory system.

```bash
python examples/holographic_demo.py
```

**What it shows:**
- Memory imprinting without backprop
- Holographic correlation
- Concept-based retrieval
- Memory efficiency

### 2. Cellular Automata (`ca_demo.py`)

Show cellular automata for text processing.

```bash
python examples/ca_demo.py
```

**What it shows:**
- Text to cellular automata
- Physics-based evolution
- Pattern recognition
- Emergent computation

### 3. Novel Architectures (`novel_architectures.py`)

Experimental architectures and ideas.

```bash
python examples/novel_architectures.py
```

**What it shows:**
- Alternative attention mechanisms
- New activation functions
- Custom layer types
- Performance comparisons

---

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── __init__.py                 # Package initialization
│
├── basic/                      # Basic functionality demos
│   ├── math_operations.py      # Matrix operations demo
│   ├── attention_demo.py       # Self-attention visualization
│   └── transformer_demo.py     # Complete transformer demo
│
├── advanced/                   # Advanced usage examples
│   ├── qwen_conversion.py      # Model conversion example
│   ├── custom_training.py      # Training without PyTorch
│   └── multi_gpu_demo.py       # Multi-GPU inference
│
├── interactive/                # Interactive demonstrations
│   ├── interactive_chat.py     # Chat interface
│   ├── realtime_demo.py        # Real-time generation
│   └── benchmark_suite.py      # Performance benchmarking
│
├── research/                   # Research-oriented examples
│   ├── holographic_demo.py     # Holographic memory demo
│   ├── ca_demo.py             # Cellular automata demo
│   └── novel_architectures.py  # Experimental architectures
│
├── data/                       # Example datasets
│   ├── sample_qa.json         # Q&A dataset for training
│   ├── test_prompts.txt       # Test prompts for evaluation
│   └── benchmark_configs.json  # Benchmark configurations
│
└── utils/                      # Helper utilities
    ├── visualization.py       # Plotting and visualization tools
    ├── data_utils.py          # Data loading and processing
    └── performance.py         # Performance measurement tools
```

---

## 🚀 Running Examples

### Prerequisites

```bash
# Install CHIMERA
pip install -e .

# Install example dependencies
pip install matplotlib seaborn plotly
```

### Running Individual Examples

```bash
# Basic examples
python -m examples.basic.math_operations
python -m examples.basic.attention_demo
python -m examples.basic.transformer_demo

# Advanced examples
python -m examples.advanced.qwen_conversion
python -m examples.advanced.custom_training

# Interactive demos
python -m examples.interactive.interactive_chat
python -m examples.interactive.benchmark_suite
```

### Running All Examples

```bash
# Run all basic examples
python scripts/run_all_examples.py --category basic

# Run performance benchmarks
python scripts/run_all_examples.py --category benchmarks

# Run with visualization
python scripts/run_all_examples.py --visualize
```

---

## 📈 Expected Performance

### Hardware Requirements

| Component | Minimum | Recommended | High-End |
|-----------|---------|-------------|----------|
| **GPU** | OpenGL 3.3 | OpenGL 4.6 | Multiple GPUs |
| **RAM** | 4GB | 16GB | 32GB+ |
| **Storage** | 1GB | 10GB | 100GB+ |

### Performance Expectations

| Operation | Intel UHD | RTX 3080 | A100 |
|-----------|-----------|----------|------|
| Matrix Mult | 15ms | 1.8ms | 0.9ms |
| Attention | 25ms | 2.1ms | 1.1ms |
| Generation | 200ms | 15ms | 8ms |

---

## 🐛 Troubleshooting

### Common Issues

**1. OpenGL Context Creation Failed**
```bash
# Update GPU drivers
# On Windows: Update via Device Manager
# On Linux: Install mesa-utils
# On macOS: Update to latest macOS
```

**2. Out of Memory Errors**
```python
# Reduce batch size or model size
model = Model(max_batch_size=1, max_seq_len=512)
```

**3. Poor Performance on Integrated Graphics**
```python
# Enable software rendering (slower but works)
import os
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
```

### Getting Help

- 📖 Check the [main documentation](../README.md)
- 🐛 Report issues on [GitHub Issues](https://github.com/chimera-ai/chimera/issues)
- 💬 Ask in [Discord community](https://discord.gg/chimera-ai)

---

## 🤝 Contributing

Want to add your own example? Here's how:

1. Create a new file in the appropriate subdirectory
2. Follow the example template structure
3. Add comprehensive documentation
4. Include performance expectations
5. Test on multiple hardware configurations

### Example Template

```python
"""
Example Title

Brief description of what this example demonstrates.

Requirements:
- Hardware: GPU with OpenGL 3.3+
- Dependencies: list any additional packages needed

Usage:
    python examples/path/to/example.py

Expected Output:
    Description of what the user should see
"""

import chimera_v3
# ... rest of your example code
```

---

**Happy exploring! 🚀**

*These examples demonstrate the revolutionary potential of OpenGL-based deep learning. Each example shows how CHIMERA eliminates the need for traditional frameworks while delivering superior performance and universal compatibility.*
