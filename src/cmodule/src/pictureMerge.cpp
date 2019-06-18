#include "pictureMerge.hpp"

#include <stdlib.h>
#include <string.h>
#include <cstdio>

#include "blendingOperators.hpp"
#include "cieLab.hpp"
#include "rgb.hpp"

void pictureMerge::printColorMatrix(double const *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    printf("%d:", i);
    for (int j = 0; j < n; ++j) {
      printf("%d: ", j);
      for (int k = 0; k < 3; ++k) {
        printf("%f ", matrix[(j + i * n) * 3 + k]);
      }
    }
    printf("\n");
  }
  printf("\n");
}

void pictureMerge::printMatrix(double const *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    printf("%d:", i);
    for (int j = 0; j < n; ++j) {
      printf("%d: %.15f ", j, matrix[j + i * n]);
    }
    printf("\n");
  }
}

void pictureMerge::mmMultSimple(int m, int n, double *A, double *weightsA,
                                double *B, double *weightsB, double *C,
                                double *weightsC) {
  bool changeA = true;
  bool changeB = true;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      struct blendingOperators::returnStruct returnValues;
      for (int k = 0; k < 3; ++k) {
        if (1.0 - A[(j + i * n) * 3] > 1e-14) {
          changeA = false;
        }
        if (1.0 - B[(j + i * n) * 3] > 1e-14) {
          changeB = false;
        }
      }
      if (changeA) {
        for (int k = 0; k < 3; ++k) {
          C[(j + i * n) * 3 + k] = A[(j + i * n) * 3 + k];
        }
        weightsC[j + i * n] = weightsA[j + i * n];
      } else if (changeB) {
        for (int k = 0; k < 3; ++k) {
          C[(j + i * n) * 3 + k] = B[(j + i * n) * 3 + k];
        }
        weightsC[j + i * n] = weightsB[j + i * n];
      } else {
        returnValues = blendingOperators::weightedPorterDuffSourceOver(
            &A[(j + i * n) * 3], weightsA[j + i * n], &B[(j + i * n) * 3],
            weightsB[j + i * n]);
        for (int k = 0; k < 3; ++k) {
          C[(j + i * n) * 3 + k] = returnValues.returnList[k];
        }
        weightsC[j + i * n] = returnValues.returnWeight;
      }
      changeA = true;
      changeB = true;
    }
  }
}

struct _indexTracker {
  double value;
  int index;
};

int _cmpfunc(const void *a, const void *b) {
  struct _indexTracker *a1 = (struct _indexTracker *)a;
  struct _indexTracker *a2 = (struct _indexTracker *)b;
  if ((*a1).value < (*a2).value)
    return -1;
  else if ((*a1).value > (*a2).value)
    return 1;
  else
    return 0;
}

int _checkIfColor(double *A, double *B) {
  bool aIsZero = true;
  bool bIsZero = true;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      if (std::abs(1.0 - A[k]) >= 1e-14) {
        aIsZero = false;
      }
      if (std::abs(1.0 - B[k]) >= 1e-14) {
        bIsZero = false;
      }
    }
  }
  if (aIsZero) return 1;
  if (bIsZero) return 2;
  return 0;
}

RGB _rgbColorspace;
CIELab _labColorspace;

void _getCieLab(double *rgb, double *lab) {
  std::vector<double> _xyz(3);
  _rgbColorspace.toXYZ(rgb, _xyz.data());
  _labColorspace.fromXYZ(_xyz.data(), lab);
}

void _getRgb(double *lab, double *rgb) {
  std::vector<double> _xyz(3);
  _labColorspace.toXYZ(lab, _xyz.data());
  _rgbColorspace.fromXYZ(_xyz.data(), rgb);
  _rgbColorspace.clamp(rgb);
}

void pictureMerge::mmMultHierarchic(int m, int n, int numberOfMatrizes,
                                    double **matrizes, double **weights,
                                    double *C, double *weightsC,
                                    const char *colorspace) {
#pragma omp parallel for
  for (int i = 0; i < m * n; ++i) {
    std::vector<double> _rgb(3 * numberOfMatrizes);
    std::vector<double> _cieLab1(3 * numberOfMatrizes);
    std::vector<double> _cieLab2(3 * numberOfMatrizes);

    // sortieren der Punkte nacht ihrer Gewichtung
    struct _indexTracker sorted_list[numberOfMatrizes];
    struct blendingOperators::returnStruct returnValues;

    struct blendingOperators::returnStruct returnLabValues;

    for (int l = 0; l < numberOfMatrizes; ++l) {
      sorted_list[l].value = weights[l][i];
      sorted_list[l].index = l;
    }

    std::qsort(sorted_list, numberOfMatrizes, sizeof(sorted_list[0]), _cmpfunc);

    int isZero = _checkIfColor(&matrizes[sorted_list[0].index][i * 3],
                               &matrizes[sorted_list[1].index][i * 3]);
    if (isZero == 1) {
      returnValues.returnList = &(matrizes[sorted_list[1].index][i * 3]);
      returnValues.returnWeight = weights[sorted_list[1].index][i];
    } else if (isZero == 2) {
      returnValues.returnList = &(matrizes[sorted_list[0].index][i * 3]);
      returnValues.returnWeight = weights[sorted_list[0].index][i];
    } else {
      if (strncmp(colorspace, "lab", 3) == 0) {
        _getCieLab(&matrizes[sorted_list[0].index][i * 3], _cieLab1.data());
        _getCieLab(&matrizes[sorted_list[1].index][i * 3], _cieLab2.data());

        returnLabValues = blendingOperators::weightedPorterDuffSourceOver(
            _cieLab1.data(), weights[sorted_list[0].index][i], _cieLab2.data(),
            weights[sorted_list[1].index][i]);
        _getRgb(returnLabValues.returnList, _rgb.data());
        returnValues.returnList = _rgb.data();
        returnValues.returnWeight = returnLabValues.returnWeight;
      } else {
        returnValues = blendingOperators::weightedPorterDuffSourceOver(
            &matrizes[sorted_list[0].index][i * 3],
            weights[sorted_list[0].index][i],
            &matrizes[sorted_list[1].index][i * 3],
            weights[sorted_list[1].index][i]);
      }
    }
    if (numberOfMatrizes > 2) {
      for (int k = 2; k < numberOfMatrizes; ++k) {
        isZero = _checkIfColor(returnValues.returnList,
                               &matrizes[sorted_list[k].index][i * 3]);
        if (isZero == 1) {
          returnValues.returnList = &(matrizes[sorted_list[k].index][i * 3]);
          returnValues.returnWeight = weights[sorted_list[k].index][i];
        } else if (isZero == 0) {
          if (strncmp(colorspace, "lab", 3) == 0) {
            _getCieLab(returnValues.returnList, _cieLab1.data() + k * 3);
            _getCieLab(&matrizes[sorted_list[k].index][i * 3],
                       _cieLab2.data() + k * 3);

            returnLabValues = blendingOperators::weightedPorterDuffSourceOver(
                _cieLab1.data() + k * 3, returnValues.returnWeight,
                _cieLab2.data() + k * 3, weights[sorted_list[k].index][i]);

            _getRgb(returnLabValues.returnList, _rgb.data());
            returnValues.returnList = _rgb.data();
            returnValues.returnWeight = returnLabValues.returnWeight;
          } else {
            returnValues = blendingOperators::weightedPorterDuffSourceOver(
                returnValues.returnList, returnValues.returnWeight,
                &matrizes[sorted_list[k].index][i * 3],
                weights[sorted_list[k].index][i]);
          }
        }
      }
    }
    for (int k = 0; k < 3; ++k) {
      C[i * 3 + k] = returnValues.returnList[k];
    }
    weightsC[i] = returnValues.returnWeight;
  }
}
