#include "rgb.hpp"
#include <cmath>

void RGB::toXYZ(const double rgb[], double xyz[]) {
  double var[3];
  for (int i = 0; i < 3; ++i) {
    if (rgb[i] > 0.04045) {
      var[i] = std::pow((rgb[i] + 0.055) / 1.055, 2.4);
    } else {
      var[i] = rgb[i] / 12.92;
    }
  }
  xyz[0] = var[0] * 0.4124 + var[1] * 0.3576 + var[2] * 0.1805;
  xyz[1] = var[0] * 0.2126 + var[1] * 0.7152 + var[2] * 0.0722;
  xyz[2] = var[0] * 0.0193 + var[1] * 0.1192 + var[2] * 0.9505;
}

void RGB::fromXYZ(const double xyz[], double rgb[]) const {
  rgb[0] = xyz[0] * 3.2406 + xyz[1] * -1.5372 + xyz[2] * -0.4986;
  rgb[1] = xyz[0] * -0.9689 + xyz[1] * 1.8758 + xyz[2] * 0.0415;
  rgb[2] = xyz[0] * 0.0557 + xyz[1] * -0.2040 + xyz[2] * 1.0570;
  for (int i = 0; i < 3; ++i) {
    if (rgb[i] > 0.0031308) {
      rgb[i] = 1.055 * std::pow(rgb[i], (1.0 / 2.4)) - 0.055;
    } else {
      rgb[i] *= 12.92;
    }
  }
}

void RGB::clamp(double rgb[]) const {
  for (int i = 0; i < 3; i++)
    if (rgb[i] > 1.0)
      rgb[i] = 1.0;
    else if (rgb[i] < 0.0)
      rgb[i] = 0.0;
}
