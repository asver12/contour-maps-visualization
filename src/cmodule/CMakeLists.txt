cmake_minimum_required(VERSION 3.1.3)

project (PICTUREMERGE)


add_subdirectory (src)
add_subdirectory (tests)

if (GTest_FOUND)
  enable_testing ()

  add_test(
      NAME runBlendingOperatorsUnitTests
      COMMAND runBlendingOperatorsUnitTests
  )

  add_test(
      NAME runPictureMergeUnitTests
      COMMAND runPictureMergeUnitTests
  )
endif()
