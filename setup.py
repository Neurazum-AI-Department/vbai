"""
Vbai - Visual Brain AI Library
Setup script for pip installation.
"""

from setuptools import setup, find_packages

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vbai",
    version="0.1.0",
    author="Neurazum",
    author_email="contact@neurazum.com",
    description="Visual Brain AI - Multi-task brain MRI analysis library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Neurazum-AI-Department/vbai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "matplotlib>=3.3.0",
            "opencv-python>=4.5.0",
            "tqdm>=4.60.0",
            "tensorboard>=2.4.0",
            "pyyaml>=5.4.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vbai-train=vbai.cli:train_cli",
            "vbai-predict=vbai.cli:predict_cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
