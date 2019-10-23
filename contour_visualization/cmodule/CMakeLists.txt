cmake_minimum_required(VERSION 3.1.3)

project (PICTUREMERGE)

include(gtest.cmake)
find_package (Threads)

add_subdirectory (src)
add_subdirectory (tests)

enable_testing ()
add_test(
  NAME runBlendingOperatorsUnitTests
  COMMAND runBlendingOperatorsUnitTests
)

add_test(
  NAME runPictureMergeUnitTests
  COMMAND runPictureMergeUnitTests
)
