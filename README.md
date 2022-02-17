# spbla

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)
[![Ubuntu](https://github.com/JetBrains-Research/spbla/actions/workflows/ubuntu.yml/badge.svg?branch=main)](https://github.com/JetBrains-Research/spbla/actions/workflows/ubuntu.yml)
[![Pages](https://github.com/JetBrains-Research/spbla/actions/workflows/docs.yml/badge.svg?branch=main)](https://jetbrains-research.github.io/spbla/)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/JetBrains-Research/spbla/blob/master/LICENSE)
[![Package](https://img.shields.io/badge/pypi%20package-1.0.0-%233776ab)](https://pypi.org/project/pyspbla/)

**spbla** is a linear Boolean algebra library primitives and operations for work with sparse matrices written for CPU,
Cuda and OpenCL platforms. The primary goal of the library is implementation, testing and profiling algorithms for
solving *formal-language-constrained problems*, such as *context-free*
and *regular* path queries with various semantics for graph databases. The library provides C-compatible API, written in
the GraphBLAS style. **The library** is shipped with python package **pyspbla** - wrapper for spbla library C API. This package exports
library features and primitives in high-level format with automated resources management and fancy syntax sugar.

* **PyPI package:** [https://pypi.org/project/pyspbla/](https://pypi.org/project/pyspbla/)
* **Tutorial:** [https://github.com/JetBrains-Research/spbla/blob/main/docs/tutorial.md](https://github.com/JetBrains-Research/spbla/blob/main/docs/tutorial.md)
* **Extended example:** [https://github.com/JetBrains-Research/spbla/blob/main/docs/getting_started.md](https://github.com/JetBrains-Research/spbla/blob/main/docs/getting_started.md)
* **Getting started:** [https://github.com/JetBrains-Research/spbla/blob/main/docs/getting_started.md](https://github.com/JetBrains-Research/spbla/blob/main/docs/getting_started.md)
* **Contributing guide:** [https://github.com/JetBrains-Research/spbla/blob/master/CONTRIBUTING.md](https://github.com/JetBrains-Research/spbla/blob/master/CONTRIBUTING.md)
* **Python Reference:** [https://jetbrains-research.github.io/spbla/pydocs/pyspbla](https://jetbrains-research.github.io/spbla/pydocs/pyspbla)
* **C API Reference:** [https://jetbrains-research.github.io/spbla/cdocs/](https://jetbrains-research.github.io/spbla/cdocs/)
* **Package source code:** [https://github.com/JetBrains-Research/spbla/tree/main/python/pyspbla](https://github.com/JetBrains-Research/spbla/tree/main/python/pyspbla)

### Features summary

- Python package for every-day tasks
- C API for performance-critical computations
- Cuda backend for computations
- OpenCL backend for computations
- Cpu (fallback) backend for computations
- Matrix creation (empty, from data, with random data)
- Matrix-matrix operations (multiplication, element-wise addition, kronecker product)
- Matrix operations (equality, transpose, reduce to vector, extract sub-matrix)
- Matrix data extraction (as lists, as list of pairs)
- Matrix syntax sugar (pretty string printing, slicing, iterating through non-zero values)
- IO (import/export matrix from/to `.mtx` file format)
- GraphViz (export single matrix or set of matrices as a graph with custom color and label settings)
- Debug (matrix string debug markers, logging)

### Platforms

- Linux based OS (tested on Ubuntu 20.04)


### Installation

Get the latest package version from PyPI package index:

```shell
$ python3 -m pip install pyspbla
```

### Simple example

Create sparse matrices, compute matrix-matrix product and print the result to the output:

```python
import pyspbla as sp

a = sp.Matrix.empty(shape=(2, 3))
a[0, 0] = True
a[1, 2] = True

b = sp.Matrix.empty(shape=(3, 4))
b[0, 1] = True
b[0, 2] = True
b[1, 3] = True
b[2, 1] = True

print(a, b, a.mxm(b), sep="\n")
```

### Performance

Sparse Boolean matrix-matrix multiplication evaluation results are listed bellow. Machine configuration: PC with Ubuntu
20.04, Intel Core i7-6700 3.40GHz CPU, DDR4 64Gb RAM, GeForce GTX 1070 GPU with 8Gb VRAM.

![time](https://github.com/JetBrains-Research/spbla/raw/main/docs/pictures/mxm-perf-time.svg?raw=true&sanitize=true)
![mem](https://github.com/JetBrains-Research/spbla/raw/main/docs/pictures/mxm-perf-mem.svg?raw=true&sanitize=true)

The matrix data is selected from the SuiteSparse Matrix Collection [link](https://sparse.tamu.edu).

| Matrix name                |     # Rows |     Nnz M | Nnz/row | Max Nnz/row |     Nnz M^2 |
|:---------------------------|-----------:|----------:|--------:|------------:|------------:|
| SNAP/amazon0312            |    400,727 | 3,200,440 |     7.9 |          10 |  14,390,544 |
| LAW/amazon-2008            |    735,323 | 5,158,388 |     7.0 |          10 |  25,366,745 |
| SNAP/web-Google            |    916,428 | 5,105,039 |     5.5 |         456 |  29,710,164 |
| SNAP/roadNet-PA            |  1,090,920 | 3,083,796 |     2.8 |           9 |   7,238,920 |
| SNAP/roadNet-TX            |  1,393,383 | 3,843,320 |     2.7 |          12 |   8,903,897 |
| SNAP/roadNet-CA            |  1,971,281 | 5,533,214 |     2.8 |          12 |  12,908,450 |
| DIMACS10/netherlands_osm   |  2,216,688 | 4,882,476 |     2.2 |           7 |   8,755,758 |

Detailed comparison is available in the full paper text at
[link](https://github.com/YaccConstructor/articles/blob/master/2021/GRAPL/Sparse_Boolean_Algebra_on_GPGPU/Sparse_Boolean_Algebra_on_GPGPU.pdf)
.

## Directory structure

```
spbla
├── .github - GitHub Actions CI setup 
├── docs - documents, text files and various helpful stuff
├── scripts - short utility programs 
├── spbla - library core source code
│   ├── include - library public C API 
│   ├── sources - source-code for implementation
│   │   ├── core - library core and state management
│   │   ├── io - logging and i/o stuff
│   │   ├── utils - auxilary class shared among modules
│   │   ├── backend - common interfaces
│   │   ├── cuda - cuda backend
│   │   ├── opencl - opencl backend
│   │   └── sequential - fallback cpu backend
│   ├── utils - testing utilities
│   └── tests - gtest-based unit-tests collection
├── python - pyspbla related sources
│   ├── pyspbla - spbla library wrapper for python (similar to pygraphblas)
│   ├── tests - regression tests for python wrapper
│   └── data - generate data for pyspbla regression tests
├── deps - project dependencies
│   ├── clbool - OpenCL based matrix operations for dcsr, csr and coo matrices
│   ├── cub - cuda utility, required for nsparse
│   ├── gtest - google test framework for unit testing
│   └── nsparse - SpGEMM implementation for csr matrices (with unified memory, configurable)
└── CMakeLists.txt - library cmake config, add this as sub-directory to your project
```

## Contributing

If you want to contribute to this project, follow our short and simple open-source
contributors [guide](./CONTRIBUTING.md). Also have a look at [code of conduct](./CODE_OF_CONDUCT.md).

## Contributors

- Egor Orachyov (Github: [EgorOrachyov](https://github.com/EgorOrachyov))
- Maria Karpenko (Github: [mkarpenkospb](https://github.com/mkarpenkospb))
- Pavel Alimov (Github : [Krekep](https://github.com/Krekep))
- Semyon Grigorev (Github: [gsvgit](https://github.com/gsvgit))

## Citation

```ignorelang
@online{spbla,
  author = {Orachyov, Egor and Karpenko, Maria and Alimov, Pavel and Grigorev, Semyon},
  title = {spbla: sparse Boolean linear algebra for CPU, Cuda and OpenCL computations},
  year = 2021,
  url = {https://github.com/JetBrains-Research/spbla},
  note = {Version 1.0.0}
}
```

## License

This project is licensed under MIT License. License text can be found in the
[license file](https://github.com/JetBrains-Research/spbla/blob/master/LICENSE.md).

## Acknowledgments <img align="right" width="15%" src="https://github.com/JetBrains-Research/spbla/raw/main/docs/pictures/jetbrains-logo.png?raw=true&sanitize=true">

This is a research project of the Programming Languages and Tools Laboratory at JetBrains-Research. Laboratory
website [link](https://research.jetbrains.org/groups/plt_lab/).