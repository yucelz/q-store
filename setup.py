"""
Q-Store Binary Distribution Setup
Compiles all source code to binary extensions for closed-source distribution
"""
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
import glob

# Custom build command to exclude .py files that have been compiled
class CustomBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        """Override to exclude .py files that have .so equivalents"""
        modules = super().find_package_modules(package, package_dir)
        # Only keep __init__.py files
        return [(pkg, mod, file) for pkg, mod, file in modules if mod == '__init__']

# Get all Python files in src/q_store to compile
def get_python_files_to_compile():
    """Get all .py files in src/q_store except __init__.py files"""
    python_files = []

    for root, dirs, files in os.walk('src/q_store'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                filepath = os.path.join(root, file)
                python_files.append(filepath)

    return python_files

# Files to compile to binary
files_to_compile = get_python_files_to_compile()

print(f"Compiling {len(files_to_compile)} Python files to binary extensions:")
for f in files_to_compile:
    print(f"  - {f}")

setup(
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Include __init__.py files and README
    package_data={
        'q_store': ['__init__.py'],
        'q_store.backends': ['__init__.py'],
        'q_store.core': ['__init__.py'],
        'q_store.ml': ['__init__.py', 'README.md'],
        'q_store.layers': ['__init__.py'],
        'q_store.layers.quantum_core': ['__init__.py'],
        'q_store.layers.classical_minimal': ['__init__.py'],
        'q_store.runtime': ['__init__.py'],
        'q_store.training': ['__init__.py'],
        'q_store.storage': ['__init__.py'],
    },

    # Compile all Python files (except __init__.py) to binary
    ext_modules=cythonize(
        files_to_compile,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'embedsignature': True,  # Keep function signatures for IDE support
        },
        # Optimization flags
        annotate=False,  # Set to True to generate HTML annotation files for debugging
        nthreads=0,  # Use all available CPU cores for parallel compilation
    ),

    cmdclass={
        'build_ext': build_ext,
        'build_py': CustomBuildPy,  # Use custom build to exclude source .py files
    },

    # Dependencies
    install_requires=[
        "numpy>=1.24.0,<3.0.0",
        "scipy>=1.10.0",
        "cirq>=1.3.0",
        "cirq-ionq>=1.3.0",
        "requests>=2.31.0",
        "zarr>=2.16.0",
        "pyarrow>=14.0.0",
        "aiohttp>=3.9.0",
    ],

    python_requires=">=3.11",

    # Prevent source distribution
    zip_safe=False,
)

