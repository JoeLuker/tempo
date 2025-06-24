#!/usr/bin/env python
"""
Setup script for TEMPO - Threshold-based Exploration with Multipath Parallel Output
"""

from setuptools import setup, find_packages
import os
import re

# Read the version from the src/__init__.py file
with open(os.path.join("src", "__init__.py"), "r") as f:
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", f.read())
    version = version_match.group(1) if version_match else "0.1.0"

# Try to use README.rst first (for PyPI), fall back to README.md
long_description = ""
content_type = "text/x-rst"
if os.path.exists("README.rst"):
    with open("README.rst", "r", encoding="utf-8") as f:
        long_description = f.read()
elif os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    content_type = "text/markdown"

# Define dependencies
install_requires = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "numpy>=1.23.0",
    "tqdm>=4.65.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "psutil>=5.9.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.3.1",
        "pytest-cov>=4.1.0",
        "black>=23.3.0",
        "isort>=5.12.0",
    ],
    # Visualization removed - not helpful for ML portfolio
    "web": [
        "uvicorn>=0.22.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
    ],
}

setup(
    name="tempo-generation",
    version=version,
    author="Joe Luker",
    author_email="github@example.com",  # Replace with actual email if available
    description="Threshold-based Exploration with Multipath Parallel Output for language model inference",
    long_description=long_description,
    long_description_content_type=content_type,
    url="https://github.com/JoeLuker/tempo",
    packages=find_packages(include=["src", "src.*"]),
    package_data={
        "src": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "tempo=run_tempo:main",
            "tempo-demo=examples.early_exit_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "nlp",
        "transformers",
        "language-model",
        "text-generation",
        "parallel-processing",
        "early-exit",
    ],
)
