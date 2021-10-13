# spbla

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)
[![Ubuntu](https://github.com/JetBrains-Research/spbla/actions/workflows/ubuntu.yml/badge.svg?branch=main)](https://github.com/JetBrains-Research/spbla/actions/workflows/ubuntu.yml)
[![Pages](https://github.com/JetBrains-Research/spbla/actions/workflows/docs.yml/badge.svg?branch=main)](https://jetbrains-research.github.io/spbla/)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/JetBrains-Research/spbla/blob/master/LICENSE)
[![Package](https://img.shields.io/badge/pypi%20package-1.0.0-%233776ab)](https://pypi.org/project/pyspbla/)

**spbla** is a linear Boolean algebra library primitives and operations for 
work with sparse matrices written for CPU, Cuda and OpenCL platforms. The primary 
goal of the library is implementation, testing and profiling algorithms for
solving *formal-language-constrained problems*, such as *context-free* 
and *regular* path queries with various semantics for graph databases.
The library provides C-compatible API, written in the GraphBLAS style.

**The library** is shipped with python package **pyspbla** - wrapper for
spbla library C API. This package exports library features and primitives 
in high-level format with automated resources management and fancy syntax sugar.

**The primary library primitive** is a sparse boolean matrix. The library provides 
the most popular operations for matrix manipulation, such as construction from
values, transpose, sub-matrix extraction, matrix-to-vector reduce, matrix-matrix
element-wise addition, matrix-matrix multiplication and Kronecker product.  

**As a fallback** library provides sequential backend for mentioned above operations
for computations on CPU side only. This backend is selected automatically
if Cuda/OpenCL compatible device is not present in the system. This can be quite handy for 
prototyping algorithms on a local computer for later running on a powerful server.  

**PyPI package web page** is available at following [link](https://pypi.org/project/pyspbla/).

### Features summary

- Python package for every-day tasks
- C API for performance-critical computations
- Cuda backend for computations
- OpenCL backend for computations
- Cpu backend for computations
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
- Windows (coming soon)
- macOS (coming soon)

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

Sparse Boolean matrix-matrix multiplication evaluation results are listed bellow.
Machine configuration: PC with Ubuntu 20.04, Intel Core i7-6700 3.40GHz CPU, DDR4 64Gb RAM, GeForce GTX 1070 GPU with 8Gb VRAM. 

![time](https://github.com/JetBrains-Research/spbla/raw/main/docs/pictures/mxm-perf-time.svg?raw=true&sanitize=true)
![mem](https://github.com/JetBrains-Research/spbla/raw/main/docs/pictures/mxm-perf-mem.svg?raw=true&sanitize=true)

The matrix data is selected from the SuiteSparse Matrix Collection [link](https://sparse.tamu.edu).

| Matrix name              | # Rows      | Nnz M       | Nnz/row   | Max Nnz/row | Nnz M^2     |
|---                       |---:         |---:         |---:       |---:         |---:         |
| SNAP/amazon0312          | 400,727     | 3,200,440   | 7.9       | 10          | 14,390,544  |
| LAW/amazon-2008          | 735,323     | 5,158,388   | 7.0       | 10          | 25,366,745  |
| SNAP/web-Google          | 916,428     | 5,105,039   | 5.5       | 456         | 29,710,164  |
| SNAP/roadNet-PA          | 1,090,920   | 3,083,796   | 2.8       | 9           | 7,238,920   |
| SNAP/roadNet-TX	       | 1,393,383   | 3,843,320   | 2.7       | 12          | 8,903,897   |
| SNAP/roadNet-CA	       | 1,971,281   | 5,533,214   | 2.8       | 12          | 12,908,450  |
| DIMACS10/netherlands_osm | 2,216,688   | 4,882,476   | 2.2       | 7           | 8,755,758   |
  
Detailed comparison is available in the full paper text at 
[link](https://github.com/YaccConstructor/articles/blob/master/2021/GRAPL/Sparse_Boolean_Algebra_on_GPGPU/Sparse_Boolean_Algebra_on_GPGPU.pdf).

## Getting started

This section gives instructions to build the library from sources.
These steps are required if you want to build library for your specific platform with custom build settings.

### Requirements

- Linux-based OS (tested on Ubuntu 20.04)
- CMake Version 3.15 or higher
- CUDA Compatible GPU device (to run Cuda computations)
- GCC Compiler 
- NVIDIA CUDA toolkit (to build Cuda backend)
- Python 3 (for `pyspbla` library)
- Git (to get source code)

### Cuda & compiler setup

> Skip this section if you want to build library with only sequential backend
> without cuda backend support.

Before the CUDA setup process, validate your system NVIDIA driver with `nvidia-smi`
command. Install required driver via `ubuntu-drivers devices` and 
`apt install <driver>` commands respectively.

The following commands grubs the required GCC compilers for the CC and CXX compiling 
respectively. CUDA toolkit, shipped in the default Ubuntu package manager, has version 
number 10 and supports only GCC of the version 8.4 or less.  

```shell script
$ sudo apt update
$ sudo apt install gcc-8 g++-8
$ sudo apt install nvidia-cuda-toolkit
$ sudo apt install nvidia-cuda-dev 
$ nvcc --version
```

If everything successfully installed, the last version command will output 
something like this:

```shell script
$ nvcc: NVIDIA (R) Cuda compiler driver
$ Copyright (c) 2005-2019 NVIDIA Corporation
$ Built on Sun_Jul_28_19:07:16_PDT_2019
$ Cuda compilation tools, release 10.1, V10.1.243
```

**Bonus Step:** In order to have CUDA support in the CLion IDE, you will have to
overwrite global alias for the `gcc` and `g++` compilers:

```shell script
$ sudo rm /usr/bin/gcc
$ sudo rm /usr/bin/g++
$ sudo ln -s /usr/bin/gcc-8 /usr/bin/gcc
$ sudo ln -s /usr/bin/g++-8 /usr/bin/g++
```

This step can be easily undone by removing old aliases and creating new one 
for the desired gcc version on your machine. Also you can safely omit this step
if you want to build library from the command line only. 

**Useful links:**
- [NVIDIA Drivers installation Ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
- [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Hello world program](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [CUDA CMake tutorial](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)

### Get the source code and run

Run the following commands in the command shell to download the repository,
make `build` directory, configure `cmake build` and run compilation process.
First of all, get the source code and project dependencies:

```shell script
$ git clone https://github.com/JetBrains-Research/spbla.git
$ cd spbla
$ git submodule update --init --recursive
```

Make the build directory and go into it:

```shell script
$ mkdir build
$ cd build
```

Configure build in Release mode with tests and run actual compilation process:

```shell script
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DSPBLA_BUILD_TESTS=ON
$ cmake --build . --target all -j `nproc`
$ bash ./scripts/run_tests_all.sh
```

By default, the following cmake options will be automatically enabled:

- `SPBLA_WITH_CUDA` - build library with actual cuda backend
- `SPBLA_WITH_OPENCL` - build library with actual cuda backend
- `SPBLA_WITH_SEQUENTIAL` - build library witt cpu based backend
- `SPBLA_WITH_TESTS` - build library unit-tests collection
- `SPBLA_WITH_CUB` - build library with bundled CUB sources, relevant for CUDA SDK 10 and earlier

> Note: in order to provide correct GCC version for CUDA sources compiling,
> you will have to provide custom paths to the CC and CXX compilers before 
> the actual compilation process as follows:
>
> ```shell script
> $ export CC=/usr/bin/gcc-8
> $ export CXX=/usr/bin/g++-8
> $ export CUDAHOSTCXX=/usr/bin/g++-8
> ```

### Python package

**Export** env variable `PYTHONPATH="/build_dir_path/python/:$PYTHONPATH"` if
you want to use `pyspbla` without installation into default python packages dir.
This variable will help python find package if you import it as `import pyspbla` in your python scripts.

#### Tests

**To run regression tests** within your build directory, open folder `/build_dir_path/python` and
run the following command:

```shell script
$ export PYTHONPATH="`pwd`:$PYTHONPATH"
$ cd tests
$ python3 -m unittest discover -v
```

**Note:** after the build process, the shared library object will be placed
inside the build directory in the folder with python wrapper `python/pyspbla/`. 
So, the wrapper will be able to automatically locate required lib file. 

#### Package config

You can configure python package by the usage of the following **optional** env variables:

- **SPBLA_PATH** - path to the compiled **spbla** library. Setup this variable, 
if you want to use your custom library build.
Setup this variable as `/path/to/the/compiled/library/libspbla.so` (actual lib name depend on target platform).

- **SPBLA_BACKEND** - string name of the preferred backend for computations. Allowed options are `default`
(default backend will be selected), `cpu`, `cuda` and `opencl`.

Following example shows how to configure these variables within Python runtime:

```python
# import os
# os.environ["SPBLA_BACKEND"] = "cpu"
# os.environ["SPBLA_BACKEND"] = "cuda"
# os.environ["SPBLA_BACKEND"] = "opencl"

# Uncomment desired line to setup selected backend before actual package import
import pyspbla as sp
```

## Usage 

The following C++ code snipped demonstrates, how library functions and
primitives can be used for the transitive closure evaluation of the directed
graph, represented as an adjacency matrix with boolean values. The transitive
closure provides info about reachable vertices in the graph:

```c++
/**
 * Performs transitive closure for directed graph
 *
 * @param A Adjacency matrix of the graph
 * @param T Reference to the handle where to allocate and store result
 *
 * @return Status on this operation
 */
spbla_Status TransitiveClosure(spbla_Matrix A, spbla_Matrix* T) {
    spbla_Matrix_Duplicate(A, T);                       /* Duplicate A to result T */

    spbla_Index total = 0;
    spbla_Index current;

    spbla_Matrix_Nvals(*T, &current);                   /* Query current nvals value */

    while (current != total) {                          /* Iterate, while new values are added */
        total = current;
        spbla_MxM(*T, *T, *T, SPBLA_HINT_ACCUMULATE);  /* T += T x T */
        spbla_Matrix_Nvals(*T, &current);
    }

    return SPBLA_STATUS_SUCCESS;
}
```

The following Python code snippet demonstrates, how the library python
wrapper can be used to compute the same transitive closure problem for the
directed graph within python environment:

```python
import pyspbla as sp

def transitive_closure(a: sp.Matrix):
    """
    Evaluates transitive closure for the provided
    adjacency matrix of the graph.

    :param a: Adjacency matrix of the graph
    :return: The transitive closure adjacency matrix
    """

    t = a.dup()                           # Duplicate matrix where to store result
    total = 0                             # Current number of values

    while total != t.nvals:
        total = t.nvals
        t.mxm(t, out=t, accumulate=True)  # t += t * t

    return t
```

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

## Acknowledgments

This is a research project of the Programming Languages and Tools Laboratory
at JetBrains-Research. Laboratory website [link](https://research.jetbrains.org/groups/plt_lab/).