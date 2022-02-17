## Getting started

This section gives instructions to build the library from sources. These steps are required if you want to build library
for your specific platform with custom build settings.

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

The following commands grubs the required GCC compilers for the CC and CXX compiling respectively. CUDA toolkit, shipped
in the default Ubuntu package manager, has version number 10 and supports only GCC of the version 8.4 or less.

```shell script
$ sudo apt update
$ sudo apt install gcc-8 g++-8
$ sudo apt install nvidia-cuda-toolkit
$ sudo apt install nvidia-cuda-dev 
$ nvcc --version
```

If everything successfully installed, the last version command will output something like this:

```shell script
$ nvcc: NVIDIA (R) Cuda compiler driver
$ Copyright (c) 2005-2019 NVIDIA Corporation
$ Built on Sun_Jul_28_19:07:16_PDT_2019
$ Cuda compilation tools, release 10.1, V10.1.243
```

**Bonus Step:** In order to have CUDA support in the CLion IDE, you will have to overwrite global alias for the `gcc`
and `g++` compilers:

```shell script
$ sudo rm /usr/bin/gcc
$ sudo rm /usr/bin/g++
$ sudo ln -s /usr/bin/gcc-8 /usr/bin/gcc
$ sudo ln -s /usr/bin/g++-8 /usr/bin/g++
```

This step can be easily undone by removing old aliases and creating new one for the desired gcc version on your machine.
Also you can safely omit this step if you want to build library from the command line only.

**Useful links:**

- [NVIDIA Drivers installation Ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
- [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Hello world program](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [CUDA CMake tutorial](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)

### Get the source code and run

Run the following commands in the command shell to download the repository, make `build` directory,
configure `cmake build` and run compilation process. First of all, get the source code and project dependencies:

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

**Export** env variable `PYTHONPATH="/build_dir_path/python/:$PYTHONPATH"` if you want to use `pyspbla` without
installation into default python packages dir. This variable will help python find package if you import it
as `import pyspbla` in your python scripts.

#### Tests

**To run regression tests** within your build directory, open folder `/build_dir_path/python` and run the following
command:

```shell script
$ export PYTHONPATH="`pwd`:$PYTHONPATH"
$ cd tests
$ python3 -m unittest discover -v
```

**Note:** after the build process, the shared library object will be placed inside the build directory in the folder
with python wrapper `python/pyspbla/`. So, the wrapper will be able to automatically locate required lib file.

#### Package config

You can configure python package by the usage of the following **optional** env variables:

- **SPBLA_PATH** - path to the compiled **spbla** library. Setup this variable, if you want to use your custom library
  build. Setup this variable as `/path/to/the/compiled/library/libspbla.so` (actual lib name depend on target platform).

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