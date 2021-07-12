# Changelog

All notable changes to this project are documented in this file.
Changes format: version name and date, minor notes, new features, fixes, what removed.

Release are available at [link](https://github.com/JetBrains-Research/spbla/releases). 

## v1.0.0 - June 8, 2021

### Summary 

**spbla** is a sparse linear Boolean algebra for Nvidia Cuda, OpenCL and CPU computations. 
Library provides C compatible API, the core of the library is written on C++ with
CUDA C/C++, CUDA Thrust and OpenCL for actual backend implementation. 
Library supports CPU backend as fallback for debugging, prototyping and running on low-end PC. 
**pyspbla** Python-package sources are shipped with the library source code. 
It provides high-level safe and efficient access to the library within Python runtime.

Python-package for Linux-based OSs is published in PyPI and available at [link](https://pypi.org/project/pyspbla/).

#### New features

- Cuda backend
- OpenCL backend
- Sequential (CPU) backend
- Sparse matrix support
- Matrix creation (empty, from data, with random data)
- Matrix-matrix operations (multiplication, element-wise addition, kronecker product)
- Matrix operations (equality, transpose, reduce to vector, extract sub-matrix)
- Matrix data extraction (as lists, as list of pairs)
- Matrix syntax sugar (pretty string printing, slicing, iterating through non-zero values)
- IO (import/export matrix from/to .mtx file format)
- GraphViz (export single matrix or set of matrices as a graph with custom color and label settings)
- Debug (matrix string debug markers, logging, operations time profiling)