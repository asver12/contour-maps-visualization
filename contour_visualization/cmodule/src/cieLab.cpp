#include "cieLab.hpp"
#include <cmath>

void CIELab::fromXYZ(const double *xyz, double *lab) const {
  double var[3];
  var[0] = xyz[0] / referencePoint[0];
  var[1] = xyz[1] / referencePoint[1];
  var[2] = xyz[2] / referencePoint[2];
  for (int i = 0; i < 3; ++i) {
    if (var[i] > 0.008856) {
      var[i] = std::pow(var[i], (1.0 / 3.0));
    } else {
      var[i] = (7.787 * var[i]) + (16.0 / 116.0);
    }
  }
  lab[0] = (116 * var[1]) - 16;
  lab[1] = 500 * (var[0] - var[1]);
  lab[2] = 200 * (var[1] - var[2]);
}

void CIELab::toXYZ(const double *lab, double *xyz) const {
  double var[3];
  var[1] = (lab[0] + 16) / 116.0;
  var[0] = lab[1] / 500.0 + var[1];
  var[2] = var[1] - lab[2] / 200.0;

  for (int i = 0; i < 3; ++i) {
    if (std::pow(var[i], 3) > 0.008856) {
      var[i] = std::pow(var[i], 3);
    } else {
      var[i] = (var[i] - 16.0 / 116.0) / 7.787;
    }
  }

  xyz[0] = var[0] * referencePoint[0];
  xyz[1] = var[1] * referencePoint[1];
  xyz[2] = var[2] * referencePoint[2];
}

const double CIELab::referencePoint[3] = {0.95047, 1., 1.08883};
