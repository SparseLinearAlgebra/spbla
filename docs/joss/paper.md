---
title: 'SPbLA: The Library of GPGPU-powered Sparse Boolean Linear Algebra Operations'
tags:
  - Python
  - C/C++
  - sparse-matrix
  - linear-algebra
  - graph-analysis
  - graph-algorithms
  - nvidia-cuda
  - opencl
authors:
  - name: Egor Orachev^[co-first author]
    orcid: 0000-0000-0000-0000
    affiliation: "1, 3"
  - name: Maria Karpenko
    affiliation: 2
  - name: Pavel Alimov
    affiliation: 1
  - name: Semyon Grigorev^[corresponding author]
    orcid: 0000-0000-0000-0000
    affiliation: "1, 3"    
affiliations:
 - name: Saint Petersburg State University
   index: 1
 - name: ITMO University
   index: 2
 - name: JetBrains Research
   index: 3
date: 29 June 2021
bibliography: paper.bib
---

# Summary

Sparse matrices are widely applicable in data analysis while the theory of 
matrix processing is well-established. There are a wide range of algorithms for 
basic operations such as matrix-matrix and matrix-vector multiplication, 
factorization, etc. To facilitate data analysis, tools, such as GraphBLAS API, 
provide a set of building blocks and allows for reducing algorithms to sparse
linear algebra operations. While GPGPU utilization for high-performance linear 
algebra is common, the high complexity of GPGPU programming makes the implementation 
of the complete set of sparse operations on GPGPU challenging. Thus, it is worth
to address this problem by focusing on a basic by still important case — sparse Boolean algebra.

# Statement of need

`SPbLA` is a sparse Boolean linear algebra primitives and operations
for GPGPU computations. It comes as stand-alone self-sufficient 
library with C API for high-performance computing with multiple backends
for NVIDIA CUDA, OpenCL and CPU-only platforms. The library is shipped
with PyPI `pyspbla` package [@pyspbla] for work within Python runtime. 
The primary library primitive is a sparse matrix of boolean values. The library 
provides the most popular operations for matrix manipulation, such as 
construction from values, transpose, sub-matrix extraction, matrix-to-vector 
reduce, matrix-matrix element-wise addition, multiplication and Kronecker product.  

The primary goal of the `SPbLA` is the implementation, testing and profiling
algorithms for solving data analysis problems, such as RDF analysis [@article:cfpq_and_rdf_analysis], 
bioinformatics [@article:rna_prediction], static code analysis of C and Java programs [@article:dyck_cfl_code_analysis] 
and evaluation of regular and CFL-reachability queries [@inproceedings:matrix_cfpq; @inbook:kronecker_cfpq_adbis]. 

Also we hope, that the library is a small step to move forward the implementation of 
the fully featured sparse linear algebra with generalization to arbitrary
monoids and searings for multi-GPU computations.

# Related tools

# Performance

# Acknowledgements

Work on the `SPbLA` project was supported by JetBrains Research 
programming languages and tool laboratory and by JetBrains company.

# References