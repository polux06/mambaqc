"""
Setup script for Quaternion Mamba-2.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mambaqc",
    version="0.1.0",
    author="Laurent",
    author_email="",
    description="Quaternion Mamba-2: A Quaternion State Space Dual Model with Cayley Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/polux06/mambaqc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "triton>=2.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "train": [
            "datasets>=2.14.0",
            "transformers>=4.35.0",
            "tensorboard>=2.14.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
        ],
    },
)
