"""
Q-Store Examples - Setup Configuration
Standalone example projects demonstrating Q-Store quantum database capabilities
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="q-store-examples",
    version="0.1.0",
    author="Q-Store Team",
    author_email="",
    description="Example projects demonstrating Q-Store quantum database capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yucelz/q-store",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core Q-Store dependency
        "q-store>=0.1.0",
        
        # Essential dependencies
        "numpy>=1.20.0",
        "python-dotenv>=0.19.0",
        
        # Optional ML/Training dependencies
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        
        # Data processing
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "full": [
            # All ML dependencies
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "peft>=0.4.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.41.0",
            "sentencepiece>=0.1.99",
            "protobuf>=3.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qstore-basic=q_store_examples.basic_example:main",
            "qstore-financial=q_store_examples.financial_example:main",
            "qstore-quickstart=q_store_examples.quantum_db_quickstart:main",
            "qstore-ml-training=q_store_examples.ml_training_example:main",
            "qstore-react-training=q_store_examples.tinyllama_react_training:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
