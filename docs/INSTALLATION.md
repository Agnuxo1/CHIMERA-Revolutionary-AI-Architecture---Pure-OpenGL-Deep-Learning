# 📦 CHIMERA Installation Guide

**Complete installation instructions for CHIMERA v3.0**

---

## 🎯 Overview

CHIMERA can be installed in several ways depending on your needs:

- **🚀 Quick Install**: Minimal setup for trying CHIMERA
- **🛠️ Development Install**: Full setup for contributors
- **🐳 Docker Install**: Containerized deployment
- **📦 PyPI Install**: When available on PyPI

---

## 🚀 Quick Install (Recommended)

### Step 1: Prerequisites

**Verify Python version:**
```bash
python --version
# Should show Python 3.8 or higher
```

**Verify GPU support:**
```bash
# Test OpenGL context creation
python -c "import moderngl; ctx = moderngl.create_standalone_context(); print('✅ OpenGL works!')"
```

### Step 2: Install Dependencies

**Core dependencies (10MB total):**
```bash
pip install moderngl numpy pillow
```

**Optional: Enhanced functionality:**
```bash
pip install matplotlib seaborn scikit-learn tqdm
```

**Optional: Model conversion (one-time only):**
```bash
pip install torch transformers
# Note: Can be uninstalled after model conversion
```

### Step 3: Clone Repository

```bash
git clone https://github.com/chimera-ai/chimera.git
cd chimera
```

### Step 4: Verify Installation

```bash
# Run basic demo
python chimera_v3/demo_pure.py

# Run examples
python examples/math_operations.py
```

---

## 🛠️ Development Installation

### Full Development Setup

```bash
# 1. Clone repository
git clone https://github.com/chimera-ai/chimera.git
cd chimera

# 2. Create virtual environment (recommended)
python -m venv chimera-env
source chimera-env/bin/activate  # On Windows: chimera-env\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install in development mode
pip install -e .

# 5. Install development tools
pip install -r requirements-dev.txt

# 6. Run tests
python -m pytest tests/

# 7. Check code style
flake8 chimera_v3/
black --check chimera_v3/
mypy chimera_v3/
```

### Development Tools Setup

```bash
# Pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# Documentation tools
pip install sphinx sphinx-rtd-theme myst-parser

# Profiling tools
pip install line_profiler memory_profiler
```

---

## 🐳 Docker Installation

### Using Pre-built Images

```bash
# Pull the latest image
docker pull chimera-ai/chimera:latest

# Run with GPU support
docker run --gpus all -p 8080:8080 chimera-ai/chimera:latest
```

### Building from Source

```bash
# Build the image
docker build -t chimera-ai .

# Run with different configurations
docker run -p 8080:8080 chimera-ai  # CPU only
docker run --gpus all -p 8080:8080 chimera-ai  # With GPU
```

**Docker Compose (recommended for development):**
```yaml
version: '3.8'
services:
  chimera:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 📦 Platform-Specific Instructions

### Windows Installation

**Prerequisites:**
```powershell
# Install Visual Studio Build Tools
# Install GPU drivers (NVIDIA/AMD/Intel)
# Enable WSL2 for better performance (optional)
```

**Installation:**
```powershell
# Using PowerShell
git clone https://github.com/chimera-ai/chimera.git
cd chimera

# Create virtual environment
python -m venv chimera-env
chimera-env\Scripts\activate

# Install dependencies
pip install moderngl numpy pillow

# Test installation
python chimera_v3/demo_pure.py
```

### Linux Installation

**Ubuntu/Debian:**
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip
sudo apt install mesa-utils  # For OpenGL utilities

# Install Python dependencies
pip3 install moderngl numpy pillow

# Test OpenGL
glxinfo | grep "OpenGL version"
```

**Arch Linux:**
```bash
# Install dependencies
sudo pacman -S python python-pip mesa

# Install Python packages
pip install moderngl numpy pillow
```

**Fedora/CentOS:**
```bash
# Install dependencies
sudo dnf install python3-devel mesa-libGL-devel

# Install Python packages
pip3 install moderngl numpy pillow
```

### macOS Installation

**Intel Macs:**
```bash
# Install dependencies
pip install moderngl numpy pillow

# Test installation
python chimera_v3/demo_pure.py
```

**Apple Silicon (M1/M2):**
```bash
# Install dependencies (includes Metal backend)
pip install moderngl numpy pillow

# May need to install additional dependencies
brew install mesa-glu

# Test installation
python chimera_v3/demo_pure.py
```

---

## 🔧 Hardware-Specific Setup

### NVIDIA GPUs

**For maximum performance:**
```bash
# Install optimal NVIDIA drivers
# Ubuntu: sudo ubuntu-drivers autoinstall
# Windows: Update via GeForce Experience

# Verify CUDA compatibility (optional)
nvidia-smi
```

**For development:**
```bash
# Install NVIDIA tools for monitoring
pip install pynvml

# Enable persistent mode
sudo nvidia-persistenced --persistence-mode
```

### AMD GPUs

**ROCm support (Linux only):**
```bash
# Install ROCm (if available)
# Ubuntu: Follow ROCm installation guide

# Test OpenGL
glxinfo | grep "OpenGL vendor"
```

### Intel GPUs

**Intel Graphics:**
```bash
# Update Intel drivers
# Ubuntu: sudo apt install intel-media-va-driver

# Verify OpenGL
glxinfo | grep "OpenGL version"
```

---

## 🚨 Troubleshooting Installation

### Common Installation Issues

**1. "moderngl failed to create context"**
```bash
# Update GPU drivers to latest version
# Check OpenGL version support
python -c "import moderngl; print(moderngl.create_standalone_context().info)"
```

**2. "ImportError: No module named 'moderngl'"**
```bash
# Install/update moderngl
pip uninstall moderngl
pip install --upgrade moderngl

# Check system dependencies
# Ubuntu: sudo apt install libgl1-mesa-glx
# Windows: Install OpenGL runtime
# macOS: Install Xcode command line tools
```

**3. "GPU memory allocation failed"**
```bash
# Reduce memory usage
export MODERNGL_MAX_TEXTURE_SIZE=2048

# Or in Python:
import os
os.environ['MODERNGL_MAX_TEXTURE_SIZE'] = '2048'
```

**4. "OpenGL extension not supported"**
```bash
# Force specific OpenGL version
export MESA_GLSL_VERSION_OVERRIDE=330
export MODERNGL_GLSL_VERSION=330
```

### Platform-Specific Fixes

**Windows:**
```powershell
# Install Microsoft Visual C++ redistributables
# Update Windows to latest version
# Enable hardware acceleration in Windows settings
```

**Linux:**
```bash
# Install 32-bit libraries if needed
sudo apt install libgl1-mesa-glx:i386

# Check GPU permissions
sudo usermod -a -G video $USER
```

**macOS:**
```bash
# Reset OpenGL preferences
defaults delete com.apple.opengl

# Update to latest macOS
softwareupdate --install -a
```

---

## ✅ Verification Tests

### Basic Functionality Test

```bash
# Test 1: OpenGL context
python -c "
import moderngl
ctx = moderngl.create_standalone_context()
print('✅ OpenGL context created successfully')
print(f'OpenGL version: {ctx.info}')
ctx.release()
"

# Test 2: Basic operations
python -c "
import numpy as np
import moderngl

ctx = moderngl.create_standalone_context()
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
print('✅ NumPy operations work')
ctx.release()
"
```

### CHIMERA-Specific Tests

```bash
# Test 3: CHIMERA imports
python -c "
try:
    import chimera_v3
    print('✅ CHIMERA imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
"

# Test 4: Demo execution
python chimera_v3/demo_pure.py
```

### Performance Benchmark

```bash
# Run performance tests
python examples/benchmark_suite.py

# Expected output (varies by hardware):
# Matrix Multiplication: 43.57× speedup vs CPU
# Self-Attention: 25.1× speedup vs CPU
```

---

## 📊 Installation Metrics

### Installation Time

| Method | Time | Difficulty | Completeness |
|--------|------|------------|--------------|
| **Quick Install** | **2-5 min** | **Easy** | **Core functionality** |
| Development Install | 10-15 min | Medium | Full development |
| Docker Install | 5-10 min | Easy | Containerized |

### Disk Space Usage

| Component | Size | Notes |
|-----------|------|-------|
| **Core dependencies** | **10MB** | moderngl, numpy, pillow |
| **Development tools** | **50MB** | pytest, black, sphinx |
| **Model files** | **100MB-2GB** | Depends on models used |
| **Example datasets** | **10MB** | Sample data for demos |

### Memory Usage

| Component | Typical Usage | Peak Usage |
|-----------|--------------|------------|
| **CHIMERA runtime** | **100-500MB** | **1-2GB** |
| **GPU memory** | **50-200MB** | **500MB-4GB** |
| **Python process** | **50-100MB** | **200-500MB** |

---

## 🔄 Updates and Maintenance

### Updating CHIMERA

```bash
# Update from Git
cd chimera
git pull origin main

# Reinstall if needed
pip install -e .

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Backup Important Files

```bash
# Models and trained weights
cp -r models/ models_backup_$(date +%Y%m%d)/

# Configuration files
cp -r configs/ configs_backup_$(date +%Y%m%d)/

# Logs and outputs
cp -r logs/ logs_backup_$(date +%Y%m%d)/
```

---

## 🎓 Learning More

After successful installation:

1. **📖 Read the main README** for complete overview
2. **🎯 Complete the Quick Start guide** for hands-on experience
3. **🔬 Explore examples** in the `examples/` directory
4. **📚 Study the architecture** in `docs/ARCHITECTURE.md`
5. **💬 Join the community** on Discord for support

---

## 📞 Getting Help

**Installation Support:**

- 📖 **Documentation**: [docs.chimera.ai](https://docs.chimera.ai)
- 💬 **Discord**: [Join community](https://discord.gg/chimera-ai)
- 🐛 **GitHub Issues**: [Report problems](https://github.com/chimera-ai/chimera/issues)
- 📧 **Email**: support@chimera.ai

**Still having issues?** Don't hesitate to reach out! The community is here to help.

---

**🎉 Congratulations! Your CHIMERA installation is complete!**

**Next steps:**
- 🚀 Run `python chimera_v3/demo_pure.py` to see it in action
- 📖 Read the [Quick Start guide](QUICK_START.md) for hands-on experience
- 💬 Join [Discord](https://discord.gg/chimera-ai) to connect with other users

**Welcome to the future of AI! 🌟**
