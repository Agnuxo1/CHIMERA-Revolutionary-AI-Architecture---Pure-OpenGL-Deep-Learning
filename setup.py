"""
CHIMERA v3.0 - Pure OpenGL Deep Learning
========================================

Setup configuration for the CHIMERA AI architecture.

This is the first deep learning framework that runs entirely on OpenGL
without requiring PyTorch, CUDA, or any traditional ML frameworks.

For more information, see: https://github.com/chimera-ai/chimera
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_dev = []
requirements_optional = []

requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                if "pytest" in line or "black" in line or "flake8" in line or "mypy" in line or "sphinx" in line:
                    requirements_dev.append(line)
                elif "torch" in line or "transformers" in line:
                    requirements_optional.append(line)
                else:
                    requirements.append(line)

setup(
    name="chimera-ai",
    version="3.0.0",
    author="Francisco Angulo de Lafuente",
    author_email="francisco.angulo@example.com",
    description="Pure OpenGL Deep Learning - Transformers Without PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chimera-ai/chimera",
    project_urls={
        "Documentation": "https://docs.chimera.ai",
        "Source": "https://github.com/chimera-ai/chimera",
        "Tracker": "https://github.com/chimera-ai/chimera/issues",
        "Community": "https://discord.gg/chimera-ai",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",
    keywords=[
        "deep-learning",
        "neural-networks",
        "transformers",
        "opengl",
        "gpu-computing",
        "machine-learning",
        "artificial-intelligence",
        "computer-vision",
        "natural-language-processing",
    ],
    packages=find_packages(include=["chimera*", "chimera_v3*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
        "conversion": requirements_optional,
        "all": requirements + requirements_dev + requirements_optional,
    },
    entry_points={
        "console_scripts": [
            "chimera-demo=chimera_v3.demo_pure:main",
            "chimera-chat=chimera_v3.chimera_holographic_chat:main",
            "chimera-setup=chimera_v3.setup_holographic_system:main",
        ],
    },
    package_data={
        "chimera_v3": [
            "shaders/*.glsl",
            "configs/*.json",
            "data/*.json",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    platforms=["Linux", "MacOS", "Windows"],
    test_suite="tests",
    tests_require=requirements_dev,
)
