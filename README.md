# DiskANN with TensorStore Backend

UW-Madison CS744, Fall 2022

![TensorStoreANN](TensorStoreANN.png)

Benefits of using TensorStore as the index storage backend:
* Shareable index files across multiple array formats with a uniform API
* Asynchronous I/O for high-throughput access
* Automatic handling of data caching
* Controlled concurrent I/O with remote storage backend

## Build

On a CloudLab Ubuntu 20.04 machine:

* Install necessary DiskANN dependencies (see original README below)
* Install `gcc` suite version >=10.x and set as default
* Install `cmake` version >= 3.24

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Note that Internet connection is required for the build, as the CMake involves Google's `FetchContent` utility, which will download `tensorstore` from our forked GitHub repo and its dependencies over the network.

## Run

For help messages:

```bash
./scripts/run.py [subcommand] -h
```

Parse Sift dataset fvecs into fbin format:

```bash
./scripts/run.py to_fbin --sift_base /mnt/ssd/data/sift-small/siftsmall --dataset /mnt/ssd/data/sift-tiny/sifttiny [--max_npts 1000]
```

Build on-disk index from learning input (may take very long):

```bash
./scripts/run.py build --dataset /mnt/ssd/data/sift-tiny/sifttiny
```

Convert on-disk index to zarr format tensors:

```bash
./scripts/run.py convert --dataset /mnt/ssd/data/sift-tiny/sifttiny
```

Run queries with different parameters:

```bash
./scripts/run.py query --dataset /mnt/ssd/data/sift-tiny/sifttiny [--k_depth 10] [--npts_to_cache 100] [--use_ts] [--ts_async] [-L 10 50 100]
```

Automated wrapper for run.py:
```bash
./scripts/run_tests.sh /mnt/ssd/data/gist/gist /mnt/ssd/result/gist
# the first argument is a path prefix of `*_learn.fbin`
# the second argument is the log directory
```

Bar graph plotting with `run.py` wrapper generated data:
```bash
./scripts/plot.py /mnt/ssd/result/gist /mnt/ssd/result/gist/plots
```


To run TensorStore with remote http server, create another node (assume IP address `10.10.1.2`) and launch a http server:

```bash
# at the parent directory of gist/
python3 -m http.server  # this will use 8000 port
```

Then in the previous node, run the script with remote address specified:

```bash
./scripts/run.py query --dataset /mnt/ssd/data/gist/gist --k_depth 10 --list_sizes 10 --use_ts --use_remote http://10.10.1.2:8000/gist/gist
```

This will load query from local and use TensorStore on `http://10.10.1.2:8000` server.

## TODO List

- [x] Converter from disk index to zarr tensors
- [x] Search path tensorstore reader integration
- [x] Allow turning on/off async I/O patterns for comparison
- [ ] Allow turning on/off tensorstore cache pool for comparison (currently sees no effect, needs further study)
- [x] Using a remote storage backend


# DiskANN - Original README

The goal of the project is to build scalable, performant, streaming and cost-effective approximate nearest neighbor search algorithms for trillion-scale vector search.
This release has the code from the [DiskANN paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf) published in NeurIPS 2019, 
the [streaming DiskANN paper](https://arxiv.org/abs/2105.09613) and improvements. 
This code reuses and builds upon some of the [code for NSG](https://github.com/ZJULearning/nsg) algorithm.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.



## Linux build:

Install the following packages through apt-get

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```

### Install Intel MKL
#### Ubuntu 20.04
```bash
sudo apt install libmkl-full-dev
```

#### Earlier versions of Ubuntu
Install Intel MKL either by downloading the [oneAPI MKL installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) or using [apt](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) (we tested with build 2019.4-070 and 2022.1.2.146).

```
# OneAPI MKL Installer
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146.sh
sudo sh l_BaseKit_p_2022.1.2.146.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
```

### Build
```bash
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
```

## Windows build:

The Windows version has been tested with Enterprise editions of Visual Studio 2022, 2019 and 2017. It should work with the Community and Professional editions as well without any changes. 

**Prerequisites:**

* CMake 3.15+ (available in VisualStudio 2019+ or from https://cmake.org)
* NuGet.exe (install from https://www.nuget.org/downloads)
    * The build script will use NuGet to get MKL, OpenMP and Boost packages.
* DiskANN git repository checked out together with submodules. To check out submodules after git clone:
```
git submodule init
git submodule update
```

* Environment variables: 
    * [optional] If you would like to override the Boost library listed in windows/packages.config.in, set BOOST_ROOT to your Boost folder.

**Build steps:**
* Open the "x64 Native Tools Command Prompt for VS 2019" (or corresponding version) and change to DiskANN folder
* Create a "build" directory inside it
* Change to the "build" directory and run
```
cmake ..
```
OR for Visual Studio 2017 and earlier:
```
<full-path-to-installed-cmake>\cmake ..
```
* This will create a diskann.sln solution. Open it from VisualStudio and build either Release or Debug configuration.
    * Alternatively, use MSBuild:
```
msbuild.exe diskann.sln /m /nologo /t:Build /p:Configuration="Release" /property:Platform="x64"
```
    * This will also build gperftools submodule for libtcmalloc_minimal dependency.
* Generated binaries are stored in the x64/Release or x64/Debug directories.

## Usage:

Please see the following pages on using the compiled code:

- [Commandline interface for building and search SSD based indices](workflows/SSD_index.md)  
- [Commandline interface for building and search in memory indices](workflows/in_memory_index.md) 
- [Commandline examples for using in-memory streaming indices](workflows/dynamic_index.md)
- To be added: Python interfaces and docker files
