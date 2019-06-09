// Include guard
#ifndef PICTUREMERGE_H_
#define PICTUREMERGE_H_

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

namespace pictureMerge {
struct picStruct {
  double *returnList;
  double *returnWeight;
};
void printColorMatrix(double const* matrix, int m, int n);
void printMatrix(double const* matrix, int m, int n);
void mmMultSimple(int m, int n, double *A, double *weightsA, double *B,
                  double *weightsB, double *C, double *weightedC);

struct picStruct mmMultSimpleReturn(int m, int n, double *A, double *weightsA, double *B,
                  double *weightsB);

void mmMultHierarchic(int m, int n, int numberOfMatrizes,
                      double **matrizes,
                      double **weights, double *C,
                      double *weightedC);
struct picStruct mmMultHierarchicReturn(int m, int n, int numberOfMatrizes,
                                        double **matrizes,
                                        double **weights);
}  // namespace pictureMerge

#ifdef __cplusplus
}
#endif

#endif // PICTUREMERGE_H_
