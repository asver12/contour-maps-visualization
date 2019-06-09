cmake_minimum_required(VERSION 3.1.3)
project(SVMBlendLibrary)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Werror -fopenmp ")


add_library(blendingOperators SHARED blendingOperators.cpp pictureMerge.cpp)

add_executable(blendTest main.cpp blendingOperators.cpp pictureMerge.cpp)
