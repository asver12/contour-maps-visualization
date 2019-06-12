#include <array>
#include <cstdio>

#include <chrono>
#include <random>

using std::chrono::duration;
using std::chrono::system_clock;
using namespace std;

#include "blendingOperators.hpp"
#include "pictureMerge.hpp"

void printColorMatrix(double const* matrix, int m, int n) {
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

void printMatrix(double const* matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    printf("%d:", i);
    for (int j = 0; j < n; ++j) {
      printf("%d: %f ", j, matrix[j + i * n]);
    }
    printf("\n");
  }
}

void fillMatrix(int m, int n, std::vector<double>* matrix,
                std::vector<double>* weightsMatrix, mt19937* gen,
                uniform_real_distribution<>* dis) {
  for (vector<double>::iterator i = matrix->begin(); i < matrix->end(); ++i) {
    *i = (*dis)(*gen);
  }
  for (vector<double>::iterator i = weightsMatrix->begin();
       i < weightsMatrix->end(); ++i) {
    *i = (*dis)(*gen);
  }
}

int main(int argc, char* argv[]) {
  double frontArray[] = {0.5, 0.5, 0.5};
  double backArray[] = {0.3, 0.3, 0.3};
  double alpha = 0.5;

  double* resultArray;
  resultArray =
      blendingOperators::porterDuffSourceOver(frontArray, backArray, alpha);

  std::printf(
      "Front: %f,%f,%f\n"
      "Back: %f,%f,%f\n"
      "Result: %f,%f,%f\n",
      frontArray[0], frontArray[1], frontArray[2], backArray[0], backArray[1],
      backArray[2], resultArray[0], resultArray[1], resultArray[2]);

  std::printf("\n");

  double weight_1 = 0.1;
  double weight_2 = 0.4;

  double* weightResultArray;
  struct blendingOperators::returnStruct returnValues;
  returnValues = blendingOperators::weightedPorterDuffSourceOver(
      frontArray, weight_1, backArray, weight_2);
  weightResultArray = returnValues.returnList;
  double weight = returnValues.returnWeight;

  std::printf(
      "Front: %f,%f,%f\n"
      "Back: %f,%f,%f\n"
      "Result: %f,%f,%f\n",
      frontArray[0], frontArray[1], frontArray[2], backArray[0], backArray[1],
      backArray[2], weightResultArray[0], weightResultArray[1],
      weightResultArray[2]);
  std::printf("The new weight is %f\n", weight);

  std::printf("Matrixaddition \n");

  // Defaultwerte der Matrizen
  int m = 3;
  int n = 2;
  if (argc == 2) {
    m = atoi(argv[1]);
  }
  if (argc == 3) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
  }

  // Erzeuge die Vektoren
  std::vector<double> a(m * n * 3), weightsA(m * n);
  std::vector<double> b(m * n * 3), weightsB(m * n);
  std::vector<double> d(m * n * 3), weightsD(m * n);
  std::vector<double> e(m * n * 3), weightsE(m * n);

  // Endmatrix
  std::vector<double> c(m * n * 3, 0), weightsC(m * n, 0);

  // Initialisierung der Zufallszahlen
  mt19937 gen(std::random_device{}());
  uniform_real_distribution<> dis(0, 1);

  // Fuelle die Vektoren
  fillMatrix(m, n, &a, &weightsA, &gen, &dis);
  printf("A\n");
  printf("Colors:\n");
  printColorMatrix(a.data(), m, n);
  printf("Weights:\n");
  printMatrix(weightsA.data(), m, n);
  printf("\n");

  fillMatrix(m, n, &b, &weightsB, &gen, &dis);
  printf("B\n");
  printf("Colors:\n");
  printColorMatrix(b.data(), m, n);
  printf("Weights:\n");
  printMatrix(weightsB.data(), m, n);
  printf("\n");

  fillMatrix(m, n, &d, &weightsD, &gen, &dis);
  printf("D\n");
  printf("Colors:\n");
  printColorMatrix(d.data(), m, n);
  printf("Weights:\n");
  printMatrix(weightsD.data(), m, n);
  printf("\n");

  fillMatrix(m, n, &e, &weightsE, &gen, &dis);
  printf("E\n");
  printf("Colors:\n");
  printColorMatrix(e.data(), m, n);
  printf("Weights:\n");
  printMatrix(weightsE.data(), m, n);
  printf("\n");
  // Test der Einfachen Berechnung
  double elapsed_seconds = 0;
  // Zeitmessung starten
  auto start = system_clock::now();
  pictureMerge::mmMultSimple(m, n, a.data(), weightsA.data(), b.data(),
                             weightsB.data(), c.data(), weightsC.data());
  // Ende der Zeitmessung
  auto end = system_clock::now();

  elapsed_seconds += duration<double>(end - start).count();
  printf("C\n");
  printf("Colors:\n");
  printColorMatrix(c.data(), m, n);
  printf("Weights:\n");
  printMatrix(weightsC.data(), m, n);
  printf("Normale Matrixaddition: %f\n", elapsed_seconds);
  printf("\n");

  // Endmatrix
  for (vector<double>::iterator i = c.begin(); i < c.end(); ++i) {
    *i = 0;
  }
  for (vector<double>::iterator i = weightsC.begin(); i < weightsC.end(); ++i) {
    *i = 0;
  }

  std::vector<double*> matrizes(4);
  matrizes[0] = a.data();
  matrizes[1] = b.data();
  matrizes[2] = d.data();
  matrizes[3] = e.data();

  std::vector<double*> weights(4);
  weights[0] = weightsA.data();
  weights[1] = weightsB.data();
  weights[2] = weightsD.data();
  weights[3] = weightsE.data();

  printf(
      "------------------------------------------------------------------------"
      "\n");
  printf(
      "--------------------------------------Hierarchic------------------------"
      "\n");
  start = system_clock::now();
  pictureMerge::mmMultHierarchic(m, n, 4, matrizes.data(), weights.data(),
                                 c.data(), weightsC.data());
  // Ende der Zeitmessung
  end = system_clock::now();

  printf("C\n");
  printf("Colors:\n");
  printColorMatrix(c.data(), m, n);
  printf("Weights:\n");
  printMatrix(weightsC.data(), m, n);
  printf("Hierarchic with 4 Matrizes: %f\n", elapsed_seconds);
  printf("\n");

  struct pictureMerge::picStruct cReturns =
      pictureMerge::mmMultHierarchicReturn(m, n, 4, matrizes.data(),
                                           weights.data());
  printColorMatrix(cReturns.returnList, m, n);
  printMatrix(cReturns.returnWeight, m, n);
  return 0;
}
