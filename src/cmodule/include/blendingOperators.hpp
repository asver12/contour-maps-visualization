// Include guard
#ifndef BLENDINGOPERATORS_H_
#define BLENDINGOPERATORS_H_

#ifdef __cplusplus
extern "C" {
#endif

namespace blendingOperators {
struct returnStruct {
  double *returnList;
  double returnWeight;
};

double *porterDuffSourceOver(double *frontRGB, double *backRGB, double alpha);
struct returnStruct weightedPorterDuffSourceOver(double *frontColor,
                                                 double weight_1,
                                                 double *backColor,
                                                 double weight_2);
}  // namespace blendingOperators

#ifdef __cplusplus
}
#endif

#endif  // BLENDINGOPERATORS_H_
