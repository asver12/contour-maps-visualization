#include <cmath>
#include <cstdio>

#include "blendingOperators.hpp"

double *blendingOperators::porterDuffSourceOver(double *frontColor,
                                                double *backColor,
                                                double alpha) {
  double *newList = new double[3];
  for (int i = 0; i < 3; ++i) {
    newList[i] = frontColor[i] * alpha + backColor[i] * (1 - alpha);
  }
  return newList;
}

struct blendingOperators::returnStruct
blendingOperators::weightedPorterDuffSourceOver(double *frontColor,
                                                double weight_1,
                                                double *backColor,
                                                double weight_2) {
  double *newList = new double[3];
  double alpha = weight_1 / (weight_1 + weight_2);
  for (int i = 0; i < 3; ++i) {
    newList[i] = frontColor[i] * alpha + backColor[i] * (1 - alpha);
  }
  double newWeight = weight_1 * alpha + weight_2 * (1 - alpha);
  struct blendingOperators::returnStruct returnValues = {newList, newWeight};
  return returnValues;
}
