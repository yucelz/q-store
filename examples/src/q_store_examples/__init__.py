"""
Q-Store Examples Package

A collection of examples demonstrating Q-Store quantum database capabilities.
"""

__version__ = "0.1.0"
__author__ = "Q-Store Team"

# Import main examples for easy access
from . import (
    basic_example,
    financial_example,
    quantum_db_quickstart,
    ml_training_example,
)

# React training examples
try:
    from . import (
        tinyllama_react_training,
        react_dataset_generator,
    )
except ImportError:
    # ML dependencies not installed
    pass

# v3.2: Quantum ML Training Examples
try:
    from . import examples_v3_2
except ImportError:
    # v3.2 examples not available
    pass

__all__ = [
    "basic_example",
    "financial_example",
    "quantum_db_quickstart",
    "ml_training_example",
    "tinyllama_react_training",
    "react_dataset_generator",
    "examples_v3_2",
]
