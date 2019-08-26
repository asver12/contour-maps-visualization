#ifndef CIELAB_HPP_
#define CIELAB_HPP_

class CIELab {
 public:
  void fromXYZ(const double *xyz, double *lab) const;
  void toXYZ(const double *lab, double *xyz) const;

 private:
  static const double referencePoint[3];
};

#endif  // CIELAB_HPP_
