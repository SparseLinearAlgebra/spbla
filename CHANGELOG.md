# Changelog

All notable changes to this project are documented in this file.
Changes format: version name and date, minor notes, new features, fixes, what removed.

Release are available at [link](https://github.com/JetBrains-Research/spbla/releases). 

## v1.0.0+zenodo - August 19, 2022

Project source code archive for JOSS publication `SPbLA: The Library of GPGPU-powered Sparse Boolean Linear Algebra Operations`

### Authors

- name: Egor Orachev   
  orcid: 0000-0002-0424-4059
  affiliations: 1, 3
- name: Maria Karpenko
  affiliations: 2
- name: Pavel Alimov
  affiliations: 1
- name: Semyon Grigorev   
  affiliations: 1, 3
  orcid: 0000-0002-7966-0698

### Affiliations

- name: Saint Petersburg State University
- name: ITMO University
- name: JetBrains Research

### Tags

- Python
- C
- C++
- sparse-matrix
- linear-algebra
- graph-analysis
- graph-algorithms
- nvidia-cuda
- opencl

### Note

spbla project source code for the version 1.0.0+ (Zenodo archive).

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