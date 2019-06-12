# PictureMerge
Uses Porter-Duff-Source-Over to combine multiple twodimensional Arrays with 3 values (floating points between 0. and 1. ) at each position. Returns an twodimensional Array. An hierarchic and a normal version is given.

## Getting Started

### Prerequisites
To compile the project cmake and gtest is needed. A way to install both on linux is:
```
sudo apt install libgtest-dev build-essential cmake
cd /usr/src/googletest
sudo cmake .
sudo cmake --build . --target install
```
Than create a new folder in cmodule and run cmake in it

```
mkdir build && cd build
cmake ..
make
```
now the library should be in the build folder.
