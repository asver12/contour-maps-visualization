#include <gtest/gtest.h>

#include "blendingOperators.hpp"

class TestFunctions : public testing::Test {
 protected:
  void compareArrays(double* vec1, double* vec2, int size) {
    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(vec1[i], vec2[i]);
    }
  }
};

/**
 * Tests if porterDuffSourceOver is calculated correctly 0.5*0.3+0.3*0.7 = 0.36
 */
TEST_F(TestFunctions, isCorrectResult) {
  double frontArray[] = {0.5, 0.7, 0.9};
  double backArray[] = {0.3, 0.2, 0.1};
  double alpha = 0.3;
  double expectedArray[] = {0.36, 0.35, 0.34};
  double* resultArray =
      blendingOperators::porterDuffSourceOver(frontArray, backArray, alpha);
  compareArrays(resultArray, &expectedArray[0], 3);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
