set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Werror -fopenmp ")

include_directories (${PROJECT_SOURCE_DIR}/include)

add_library(blendingOperators SHARED blendingOperators.cpp pictureMerge.cpp)

add_executable(blendTest main.cpp blendingOperators.cpp pictureMerge.cpp)
