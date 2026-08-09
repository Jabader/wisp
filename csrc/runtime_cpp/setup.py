"""
WISP C++ Runtime Extension Build Script

Builds the high-performance C++ runtime extension for Intel Xeon Gold processors.
Optimizations include:
- NUMA-aware memory allocation
- AVX-512 vectorization
- Thread affinity for physical cores
- Lock-free data structures

Usage:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import os
import sys

# Compiler flags optimized for Intel Xeon Gold (Ice Lake-SP)
common_flags = [
    '-O3',                      # Maximum optimization
    '-march=native',            # Target local CPU (enables AVX-512 on Xeon Gold)
    '-mavx512f',                # AVX-512 Foundation
    '-mavx512vl',               # AVX-512 Vector Length
    '-mavx512bw',               # AVX-512 Byte and Word
    '-mavx512dq',               # AVX-512 Doubleword and Quadword
    '-mavx512vnni',             # AVX-512 Vector Neural Network Instructions
    '-fopenmp',                 # OpenMP support
    '-std=c++17',               # C++17 standard
    '-Wall',                    # All warnings
    '-Wextra',                  # Extra warnings
    '-Wno-unused-parameter',    # Suppress unused parameter warnings
]

# Platform-specific flags
if sys.platform == 'linux':
    common_flags.extend([
        '-fPIC',                        # Position-independent code
        '-D_GLIBCXX_USE_CXX11_ABI=0',   # Compatibility with older GCC
    ])
    
    # Linker flags
    link_flags = [
        '-lnuma',                       # NUMA library
        '-fopenmp',                     # OpenMP runtime
        '-lpthread',                    # POSIX threads
    ]
else:
    link_flags = []

# Define the extension module
wisp_runtime_cpp = Extension(
    'wisp_runtime_cpp',
    sources=[
        'wisp_runtime_cpp.cpp',
    ],
    include_dirs=[
        '.',
    ],
    extra_compile_args=common_flags,
    extra_link_args=link_flags,
    language='c++',
)

setup(
    name='wisp_runtime_cpp',
    version='0.1.0',
    description='WISP C++ Runtime Extension for high-performance token generation',
    ext_modules=[wisp_runtime_cpp],
    install_requires=[
        'numpy',
        'torch',
    ],
)
