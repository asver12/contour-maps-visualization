#include "pictureMerge.hpp"

#include <stdlib.h>
#include <cstdio>

#include "blendingOperators.hpp"

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

struct pictureMerge::picStruct pictureMerge::mmMultSimpleReturn(
    int m, int n, double *A, double *weightsA, double *B, double *weightsB) {
  // Endmatrix
  std::vector<double> c(m * n * 3, 0);
  std::vector<double> weightsC(m * n, 0);
  pictureMerge::printColorMatrix(A, m, n);
  pictureMerge::printColorMatrix(B, m, n);
  pictureMerge::printMatrix(weightsA, m, n);
  pictureMerge::printMatrix(weightsB, m, n);
  pictureMerge::mmMultSimple(m, n, A, weightsA, B, weightsB, c.data(),
                             weightsC.data());
  printf("Result-Weights\n");
  pictureMerge::printMatrix(weightsC.data(), m, n);
  struct pictureMerge::picStruct returnValues = {c.data(), weightsC.data()};
  return returnValues;
}

struct indexTracker {
  double value;
  int index;
};

int cmpfunc(const void *a, const void *b) {
  struct indexTracker *a1 = (struct indexTracker *)a;
  struct indexTracker *a2 = (struct indexTracker *)b;
  if ((*a1).value < (*a2).value)
    return -1;
  else if ((*a1).value > (*a2).value)
    return 1;
  else
    return 0;
}

int checkIfColor(double *A, double *B) {
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

void pictureMerge::mmMultHierarchic(int m, int n, int numberOfMatrizes,
                                    double **matrizes, double **weights,
                                    double *C, double *weightsC) {
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // sortieren der Punkte nacht ihrer Gewichtung
      struct indexTracker sorted_list[numberOfMatrizes];
      struct blendingOperators::returnStruct returnValues;

      for (int l = 0; l < numberOfMatrizes; ++l) {
        sorted_list[l].value = weights[l][j + i * n];
        sorted_list[l].index = l;
      }
      std::qsort(sorted_list, numberOfMatrizes, sizeof(sorted_list[0]),
                 cmpfunc);

      int isZero =
          checkIfColor(&matrizes[sorted_list[0].index][(j + i * n) * 3],
                       &matrizes[sorted_list[1].index][(j + i * n) * 3]);
      if (isZero == 1) {
        returnValues.returnList =
            &(matrizes[sorted_list[1].index][(j + i * n) * 3]);
        returnValues.returnWeight = weights[sorted_list[1].index][j + i * n];
      } else if (isZero == 2) {
        returnValues.returnList =
            &(matrizes[sorted_list[0].index][(j + i * n) * 3]);
        returnValues.returnWeight = weights[sorted_list[0].index][j + i * n];
      } else {
        returnValues = blendingOperators::weightedPorterDuffSourceOver(
            &matrizes[sorted_list[0].index][(j + i * n) * 3],
            weights[sorted_list[0].index][j + i * n],
            &matrizes[sorted_list[1].index][(j + i * n) * 3],
            weights[sorted_list[1].index][j + i * n]);
      }
      if (numberOfMatrizes > 2) {
        for (int k = 2; k < numberOfMatrizes; ++k) {
          isZero =
              checkIfColor(returnValues.returnList,
                           &matrizes[sorted_list[k].index][(j + i * n) * 3]);
          if (isZero == 1) {
            returnValues.returnList =
                &(matrizes[sorted_list[k].index][(j + i * n) * 3]);
            returnValues.returnWeight =
                weights[sorted_list[k].index][j + i * n];
          } else if (isZero == 0) {
            returnValues = blendingOperators::weightedPorterDuffSourceOver(
                returnValues.returnList, returnValues.returnWeight,
                &matrizes[sorted_list[k].index][(j + i * n) * 3],
                weights[sorted_list[k].index][j + i * n]);
          }
        }
      }
      for (int k = 0; k < 3; ++k) {
        C[(j + i * n) * 3 + k] = returnValues.returnList[k];
      }
      weightsC[j + i * n] = returnValues.returnWeight;
    }
  }
}

struct pictureMerge::picStruct pictureMerge::mmMultHierarchicReturn(
    int m, int n, int numberOfMatrizes, double **matrizes, double **weights) {
  // Endmatrix
  std::vector<double> C(m * n * 3, 0), weightsC(m * n, 0);
  // pictureMerge::mmMultHierarchic(m, n, numberOfMatrizes, matrizes, weights,
  //                                C.data(), weightsC.data());
  struct pictureMerge::picStruct returnValues = {C.data(), weightsC.data()};
  return returnValues;
}
