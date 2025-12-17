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
    name="q-store",
    version="3.4.3",
    author="Yucel Zengin",
    author_email="yucelz@gmail.com",
    description="Quantum-Native Database with ML Capabilities (AGPLv3 / Commercial License)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yucelz/q-store",
    license="AGPLv3",

    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Include __init__.py files and README
    package_data={
        'q_store': ['__init__.py'],
        'q_store.backends': ['__init__.py'],
        'q_store.core': ['__init__.py'],
        'q_store.ml': ['__init__.py', 'README.md'],
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
        "torch>=2.0.0",
    ],

    python_requires=">=3.11",

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Database",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
    ],

    # Prevent source distribution
    zip_safe=False,
)

