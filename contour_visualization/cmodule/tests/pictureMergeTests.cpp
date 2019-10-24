#include <gtest/gtest.h>

#include "pictureMerge.hpp"

class TestFunctions : public testing::Test {
 protected:
  void compareValues(double* resultValues, double* expectedValues, int size) {
    for (int i = 0; i < size; ++i) {
      EXPECT_NEAR(resultValues[i], expectedValues[i], 0.0001);
    }
  }
  // for readability reasons. So it becomes more cleare where the error is
  void compareWeights(double* resultWeights, double* expectedWeights,
                      int size) {
    for (int i = 0; i < size; ++i) {
      EXPECT_NEAR(resultWeights[i], expectedWeights[i], 0.0001);
    }
  }
};

TEST_F(TestFunctions, hierarchicIsAllTheSame) {
  char colorspace[] = "rgb";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};

  std::vector<double*> matrizes(4);
  matrizes[0] = &a[0];
  matrizes[1] = &b[0];

  double aWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  double bWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double*> weights(4);
  weights[0] = &aWeights[0];
  weights[1] = &bWeights[0];

  double expectedValues[] = {.2, .2, .2, .2, .2, .2, .2, .2, .2,
                             .2, .2, .2, .2, .2, .2, .2, .2, .2};

  double expectedWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultHierarchic(m, n, 2, matrizes.data(), weights.data(),
                                 resultValues.data(), resultWeights.data(),
                                 &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, hierarchicIsCorrectResultInRgb) {
  char colorspace[] = "rgb";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};
  double c[] = {.3, .6, .9, .9, .6, .3, .3, .6, .9,
                .9, .6, .3, .3, .6, .9, .9, .6, .3};
  double d[] = {.4, .4, .4, .4, .4, .4, .4, .4, .4,
                .4, .4, .4, .4, .4, .4, .4, .4, .4};

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
      .352102885, .422106361, .492109837, .237243657, .246054918, .25486618,
      .325,       .4,         .475,       .50132082,  .422106361, .342891901,
      .246969697, .245454545, .243939394, .521212121, .43030303,  .339393939};
  double expectedWeights[] = {0.3428919, 0.3428919,  0.1,
                              0.6857838, 0.18484848, 0.18484848};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultHierarchic(m, n, 4, matrizes.data(), weights.data(),
                                 resultValues.data(), resultWeights.data(),
                                 &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, hierarchicIsCorrectResultInLab) {
  char colorspace[] = "lab";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};
  double c[] = {.3, .6, .9, .9, .6, .3, .3, .6, .9,
                .9, .6, .3, .3, .6, .9, .9, .6, .3};
  double d[] = {.4, .4, .4, .4, .4, .4, .4, .4, .4,
                .4, .4, .4, .4, .4, .4, .4, .4, .4};

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
      0.38665317, 0.41806062, 0.48362828, 0.26698167, 0.24183036, 0.25869764,
      0.36174919, 0.39463611, 0.46523667, 0.50871781, 0.41928577, 0.34742087,
      0.28254894, 0.24144769, 0.23864799, 0.52935999, 0.42698403, 0.3453761};
  double expectedWeights[] = {0.3428919, 0.3428919,  0.1,
                              0.6857838, 0.18484848, 0.18484848};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultHierarchic(m, n, 4, matrizes.data(), weights.data(),
                                 resultValues.data(), resultWeights.data(),
                                 &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaSumIsAllTheSame) {
  char colorspace[] = "rgb";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};

  std::vector<double*> matrizes(4);
  matrizes[0] = &a[0];
  matrizes[1] = &b[0];

  double aWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  double bWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double*> weights(4);
  weights[0] = &aWeights[0];
  weights[1] = &bWeights[0];

  double expectedValues[] = {.2, .2, .2, .2, .2, .2, .2, .2, .2,
                             .2, .2, .2, .2, .2, .2, .2, .2, .2};

  double expectedWeights[] = {1., 1., 1., 1., 1., 1.};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultSumHierarchic(m, n, 2, matrizes.data(), weights.data(),
                                    resultValues.data(), resultWeights.data(),
                                    &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaSumIsCorrectResultInRgb) {
  char colorspace[] = "rgb";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};
  double c[] = {.3, .6, .9, .9, .6, .3, .3, .6, .9,
                .9, .6, .3, .3, .6, .9, .9, .6, .3};
  double d[] = {.4, .4, .4, .4, .4, .4, .4, .4, .4,
                .4, .4, .4, .4, .4, .4, .4, .4, .4};

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
  double expectedValues[] = {0.320000, 0.4,      0.480000, 0.350000, 0.300000,
                             0.25,     0.275,    0.35,     0.425000, 0.5,
                             0.4,      0.300000, 0.25,     0.300000, 0.350000,
                             0.5,      0.4,      0.3};
  double expectedWeights[] = {1., 1., 1., 1., 1., 1.};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultSumHierarchic(m, n, 4, matrizes.data(), weights.data(),
                                    resultValues.data(), resultWeights.data(),
                                    &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaSumIsCorrectResultInLab) {
  char colorspace[] = "lab";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};
  double c[] = {.3, .6, .9, .9, .6, .3, .3, .6, .9,
                .9, .6, .3, .3, .6, .9, .9, .6, .3};
  double d[] = {.4, .4, .4, .4, .4, .4, .4, .4, .4,
                .4, .4, .4, .4, .4, .4, .4, .4, .4};

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
      0.3659244242109919, 0.393615328746744,   0.467321553175901,
      0.369449123232154,  0.2932236501620373,  0.25531134299009534,
      0.3200961544945721, 0.3426009902655126,  0.4121437912162032,
      0.5068872869060976, 0.3948244369052701,  0.3042045280500679,
      0.2924666308329431, 0.29346606182742774, 0.3395539101309194,
      0.5089383085193616, 0.39405969260455453, 0.30594308308552687};
  double expectedWeights[] = {1., 1., 1., 1., 1., 1.};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultSumHierarchic(m, n, 4, matrizes.data(), weights.data(),
                                    resultValues.data(), resultWeights.data(),
                                    &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaQuadIsAllTheSame) {
  char colorspace[] = "rgb";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};

  std::vector<double*> matrizes(4);
  matrizes[0] = &a[0];
  matrizes[1] = &b[0];

  double aWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  double bWeights[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double*> weights(4);
  weights[0] = &aWeights[0];
  weights[1] = &bWeights[0];

  double expectedValues[] = {.2, .2, .2, .2, .2, .2, .2, .2, .2,
                             .2, .2, .2, .2, .2, .2, .2, .2, .2};

  double expectedWeights[] = {1., 1., 1., 1., 1., 1.};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultQuadraticHierarchic(m, n, 2, matrizes.data(),
                                          weights.data(), resultValues.data(),
                                          resultWeights.data(), &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaQuadIsCorrectResultInRgb) {
  char colorspace[] = "rgb";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};
  double c[] = {.3, .6, .9, .9, .6, .3, .3, .6, .9,
                .9, .6, .3, .3, .6, .9, .9, .6, .3};
  double d[] = {.4, .4, .4, .4, .4, .4, .4, .4, .4,
                .4, .4, .4, .4, .4, .4, .4, .4, .4};

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
  double expectedValues[] = {0.346666, 0.426666, 0.506666, 0.276666, 0.259999,
                             0.243333, 0.275,    0.35,     0.425000, 0.526666,
                             0.426666, 0.326666, 0.229999, 0.26,     0.29,
                             0.56,     0.440000, 0.320000};
  double expectedWeights[] = {1., 1., 1., 1., 1., 1.};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultQuadraticHierarchic(m, n, 4, matrizes.data(),
                                          weights.data(), resultValues.data(),
                                          resultWeights.data(), &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaQuadIsCorrectResultInLab) {
  char colorspace[] = "lab";
  int m = 2;
  int n = 3;
  double a[] = {.1, .2, .3, .1, .2, .3, .1, .2, .3,
                .1, .2, .3, .1, .2, .3, .1, .2, .3};
  double b[] = {.3, .2, .1, .3, .2, .1, .3, .2, .1,
                .3, .2, .1, .3, .2, .1, .3, .2, .1};
  double c[] = {.3, .6, .9, .9, .6, .3, .3, .6, .9,
                .9, .6, .3, .3, .6, .9, .9, .6, .3};
  double d[] = {.4, .4, .4, .4, .4, .4, .4, .4, .4,
                .4, .4, .4, .4, .4, .4, .4, .4, .4};

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
  double expectedValues[] = {0.388530, 0.421862, 0.496115, 0.303674, 0.254681,
                             0.248184, 0.320096, 0.342600, 0.412143, 0.532751,
                             0.423250, 0.331142, 0.269186, 0.255136, 0.2832292,
                             0.567235, 0.435485, 0.3270027};
  double expectedWeights[] = {1., 1., 1., 1., 1., 1.};
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultQuadraticHierarchic(m, n, 4, matrizes.data(),
                                          weights.data(), resultValues.data(),
                                          resultWeights.data(), &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

TEST_F(TestFunctions, alphaQuadLiveLab) {
  char colorspace[] = "lab";
  int m = 5;
  int n = 5;
  double a[] = {
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 0.031373, 0.188235, 0.419608, 0.419608, 0.682353, 0.839216,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 0.419608, 0.682353, 0.839216, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000};
  double b[] = {
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      0.992157, 0.552941, 0.235294, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.992157, 0.552941,
      0.235294, 0.498039, 0.152941, 0.015686, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000};
  double c[] = {
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      0.454902, 0.768627, 0.462745, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 0.000000, 0.266667, 0.105882, 0.454902, 0.768627, 0.462745,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000};
  double d[] = {
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.619608, 0.603922,
      0.784314, 0.247059, 0.000000, 0.490196, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 0.619608, 0.603922, 0.784314, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000};

  std::vector<double*> matrizes(4);
  matrizes[0] = &a[0];
  matrizes[1] = &b[0];
  matrizes[2] = &c[0];
  matrizes[3] = &d[0];

  double aWeights[] = {
      9.838381640205877e-13, 4.835337152425476e-09,  1.6012437517522594e-07,
      3.572857747651543e-08, 5.3715743687529217e-11, 4.835337152425476e-09,
      2.376456436908139e-05, 0.0007869742896835666,  0.00017559769928979086,
      2.640004633089621e-07, 1.6012437517522594e-07, 0.0007869742896835666,
      0.02606100928274216,   0.0058149971748879745,  8.742494659780448e-06,
      3.572857747651543e-08, 0.00017559769928979086, 0.0058149971748879745,
      0.0012975012508954218, 1.9507142335336037e-06, 5.3715743687529217e-11,
      2.640004633089621e-07, 8.742494659780448e-06,  1.9507142335336037e-06,
      2.93278023299362e-09};
  double bWeights[] = {
      2.93278023299362e-09,  1.9507142335336037e-06, 8.742494659780448e-06,
      2.640004633089621e-07, 5.3715743687529217e-11, 1.9507142335336037e-06,
      0.0012975012508954218, 0.0058149971748879745,  0.00017559769928979086,
      3.572857747651543e-08, 8.742494659780448e-06,  0.0058149971748879745,
      0.02606100928274216,   0.0007869742896835666,  1.6012437517522594e-07,
      2.640004633089621e-07, 0.00017559769928979086, 0.0007869742896835666,
      2.376456436908139e-05, 4.835337152425476e-09,  5.3715743687529217e-11,
      3.572857747651543e-08, 1.6012437517522594e-07, 4.835337152425476e-09,
      9.838381640205877e-13};
  double cWeights[] = {
      5.3715743687529217e-11, 2.640004633089621e-07,  8.742494659780448e-06,
      1.9507142335336037e-06, 2.93278023299362e-09,   3.572857747651543e-08,
      0.00017559769928979086, 0.0058149971748879745,  0.0012975012508954218,
      1.9507142335336037e-06, 1.6012437517522594e-07, 0.0007869742896835666,
      0.02606100928274216,    0.0058149971748879745,  8.742494659780448e-06,
      4.835337152425476e-09,  2.376456436908139e-05,  0.0007869742896835666,
      0.00017559769928979086, 2.640004633089621e-07,  9.838381640205877e-13,
      4.835337152425476e-09,  1.6012437517522594e-07, 3.572857747651543e-08,
      5.3715743687529217e-11};
  double dWeights[] = {
      5.3715743687529217e-11, 3.572857747651543e-08, 1.6012437517522594e-07,
      4.835337152425476e-09,  9.838381640205877e-13, 2.640004633089621e-07,
      0.00017559769928979086, 0.0007869742896835666, 2.376456436908139e-05,
      4.835337152425476e-09,  8.742494659780448e-06, 0.0058149971748879745,
      0.02606100928274216,    0.0007869742896835666, 1.6012437517522594e-07,
      1.9507142335336037e-06, 0.0012975012508954218, 0.0058149971748879745,
      0.00017559769928979086, 3.572857747651543e-08, 2.93278023299362e-09,
      1.9507142335336037e-06, 8.742494659780448e-06, 2.640004633089621e-07,
      5.3715743687529217e-11};

  std::vector<double*> weights(4);
  weights[0] = &aWeights[0];
  weights[1] = &bWeights[0];
  weights[2] = &cWeights[0];
  weights[3] = &dWeights[0];
  double expectedValues[] = {
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      0.771003, 0.677566, 0.350266, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.846531, 0.578908,
      0.521642, 0.291863, 0.180913, 0.266393, 0.462052, 0.725269, 0.654513,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 0.532141, 0.644325, 0.811641, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
      1.000000, 1.000000, 1.000000, 1.000000, 1.000000};
  double expectedWeights[] = {
      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
  };
  std::vector<double> resultValues(m * n * 3, 0), resultWeights(m * n, 0);

  pictureMerge::mmMultQuadraticHierarchic(m, n, 4, matrizes.data(),
                                          weights.data(), resultValues.data(),
                                          resultWeights.data(), &colorspace[0]);
  compareValues(resultValues.data(), &expectedValues[0], m * n * 3);
  compareWeights(resultWeights.data(), &expectedWeights[0], m * n);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
