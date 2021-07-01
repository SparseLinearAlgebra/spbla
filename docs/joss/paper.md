---
title: 'SPbLA: The Library of GPGPU-powered Sparse Boolean Linear Algebra Operations'
tags:
  - Python
  - C
  - C++
  - sparse-matrix
  - linear-algebra
  - graph-analysis
  - graph-algorithms
  - nvidia-cuda
  - opencl
authors:
  - name: Egor Orachev
    orcid: 0000-0002-0424-4059
    affiliation: "1, 3"
  - name: Maria Karpenko
    affiliation: 2
  - name: Pavel Alimov
    affiliation: 1
  - name: Semyon Grigorev
    orcid: 0000-0002-7966-0698
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

`SPbLA` is a sparse Boolean linear algebra primitives and operations
for GPGPU computations. It comes as stand-alone self-sufficient 
library with C API for high-performance computing with multiple backends
for NVIDIA CUDA, OpenCL and CPU-only platforms. The library is shipped
with PyPI `pyspbla` package [@pyspbla] for work within Python runtime. 
The primary library primitive is a sparse matrix of boolean values. The library 
provides the most popular operations for matrix manipulation, such as 
construction from values, transpose, sub-matrix extraction, matrix-to-vector 
reduce, matrix-matrix element-wise addition, multiplication and Kronecker product.  


# Statement of need

Sparse matrices are widely applicable in data analysis and `GraphBLAS API` provides a set of unified building linear algebra based blocks for reducing data analysis algorithms to sparse linear algebra operations. While GPGPU utilization for high-performance linear algebra is common, the high complexity of GPGPU programming makes the implementation of the complete set of sparse operations on GPGPU challenging. Thus, it is worth addressing this problem by focusing on a basic but still important case — sparse Boolean algebra.

The primary goal of the `SPbLA` is to provide a base for the implementation, testing and profiling high-performance algorithms for solving data analysis problems, such as RDF analysis [@article:cfpq_and_rdf_analysis], RNA secondary structure analysis [@article:rna_prediction], static program code analysis (such as points-to or alias analysis) [@article:dyck_cfl_code_analysis] and evaluation of regular and CFL-reachability queries [@inproceedings:matrix_cfpq; @inbook:kronecker_cfpq_adbis]. 

Thus we can offload different language-constrained path querying related problems, and other problems that can be reduced to manipulation of boolean matrices, to GPGPU in a uniform way. 

Moreover, real data analysis leads to huge matrix processing that can not be efficiently handled on a single GPGPU. Thus the creation of the library which supports multi-GPU and out-of-VRAM computations helps to create an efficient solution for a wide range of applied problems. The creation of such a solution is an open problem while ad-hoc solutions exist in specific areas. We propose an SPbLA as a base for such a solution.

Also, we hope, that the library is a small step to move forward the implementation of the fully-featured sparse linear algebra as specified in GrpahBLAS multi-GPU computations.


# Related tools

`GraphBLAS API` [@paper:graphblas_foundations] is one of the first standards, that
formalize the mathematical building blocks in the form of the programming interface
for implementing algorithms in the language of the linear algebra. 
`SuiteSparse` [@article:suite_sparse_for_graph_problems] is a reference implementations
of the `GraphBLAS API` for CPU computation. It is mature and fully featured library
with number of bindings for other programming languages, such `pygraphblas` [@pygraphblas] 
for Python programming.

GPGPU utilization for data analysis and for linear algebra operations is a promising 
way to high-performance data analysis because GPGPU is much more powerful in parallel
data processing. However, GPGPU programming is still challenging.
Best to our knowledge, the is no complete GraphBLAS API implementation for GPGPU
computations, except `GraphBLAST` [@yang2019graphblast], which is currently in the
active development. Some work is also done to move SuiteSparse forward GPGPU computations.
OpenCL and portability?

The sparsity of data introduces issues with load balancing, irregular data access, 
thus sparsity complicates the implementation of high-performance algorithms for 
sparse linear algebra on GPGPU even more. There is number of open-source libraries,
which implement independently different sparse formats and operations.
Thus, there are no sparse linear algebra libraries based on the state-of-the-art algorithms.
Libraries such as `cuSPARSE` [@net:cusparse_docs], `bhSPARSE` [@10.1016/j.jpdc.2015.06.010], 
`clSPARSE` [@10.1145/2909437.2909442] and `CUSP` [@net:cusplibrary] have limited type 
and operators customization features with major focus on numerical types only.

# Performance

# Future research

# Acknowledgements

Work on the `SPbLA` project was supported by a grant from JetBrains Research.

# References