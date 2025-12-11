"""
Setup script for Q-Store quantum database
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="q-store",
    version="1.0.0",
    author="Q-Store Contributors",
    author_email="contact@example.com",
    description="Quantum-Native Database Architecture leveraging quantum mechanical properties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yucelz/q-store",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "cirq>=1.3.0",
        "cirq-ionq>=1.3.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "backends": [
            "pinecone-client>=3.0.0",
            "pgvector>=0.2.0",
            "qdrant-client>=1.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
