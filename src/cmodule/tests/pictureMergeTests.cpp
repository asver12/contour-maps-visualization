#include <gtest/gtest.h>

#include "pictureMerge.hpp"

class TestFunctions : public testing::Test {
 protected:
  void compareArrays(double* resultVector, double* expectedVector, int size) {
    for (int i = 0; i < size; ++i) {
      EXPECT_FLOAT_EQ(resultVector[i], expectedVector[i]);
    }
  }
};

TEST_F(TestFunctions, isAllTheSame) {
  int m = 2;
  int n = 3;

  double a[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  double b[] = {3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1};

  std::vector<double*> matrizes(4);
  matrizes[0] = &a[0];
  matrizes[1] = &b[0];

  double aWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  double bWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double*> weights(4);
  weights[0] = &aWeights[0];
  weights[1] = &bWeights[0];

  double expectedValues[] = {2., 2., 2., 2., 2., 2., 2., 2., 2.,
                             2., 2., 2., 2., 2., 2., 2., 2., 2.};

  double expectedWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultHierarchic(m, n, 2, matrizes.data(), weights.data(),
                                 resultValues.data(), resultWeights.data());
  compareArrays(resultValues.data(), &expectedValues[0], m * n * 3);
  compareArrays(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, isCorrectResult) {
  int m = 2;
  int n = 3;
  double a[] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  double b[] = {3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1};
  double c[] = {3, 6, 9, 9, 6, 3, 3, 6, 9, 9, 6, 3, 3, 6, 9, 9, 6, 3};
  double d[] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

  std::vector<double*> matrizes(4);
  matrizes[0] = &a[0];
  matrizes[1] = &b[0];
  matrizes[2] = &c[0];
  matrizes[3] = &d[0];

  double aWeights[] = {0.1, 0.4, 0.1, 0.2, 0.2, 0.1};
  double bWeights[] = {0.2, 0.3, 0.1, 0.4, 0.2, 0.1};
  double cWeights[] = {0.3, 0.2, 0.1, 0.6, 0.1, 0.2};
  double dWeights[] = {0.4, 0.1, 0.1, 0.8, 0.1, 0.2};

  std::vector<double*> weights(4);
  weights[0] = &aWeights[0];
  weights[1] = &bWeights[0];
  weights[2] = &cWeights[0];
  weights[3] = &dWeights[0];
  double expectedValues[] = {
      3.52102885, 4.22106361, 4.92109837, 2.37243657, 2.46054918, 2.5486618,
      3.25,       4.,         4.75,       5.0132082,  4.22106361, 3.42891901,
      2.46969697, 2.45454545, 2.43939394, 5.21212121, 4.3030303,  3.39393939};
  double expectedWeights[] = {0.3428919, 0.3428919,  0.1,
                              0.6857838, 0.18484848, 0.18484848};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultHierarchic(m, n, 4, matrizes.data(), weights.data(),
                                 resultValues.data(), resultWeights.data());
  compareArrays(resultValues.data(), &expectedValues[0], m * n * 3);
  compareArrays(resultWeights.data(), &expectedWeights[0], m * n);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
