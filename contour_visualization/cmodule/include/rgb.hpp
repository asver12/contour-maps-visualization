#ifndef RGB_HPP_
#define RGB_HPP_

#include <vector>

class RGB {
 public:
  void toXYZ(const double rgb[], double xyz[]);
  void fromXYZ(const double xyz[], double rgb[]) const;
  void clamp(double rgb[]) const;
};

#endif  // RGB_HPP_
