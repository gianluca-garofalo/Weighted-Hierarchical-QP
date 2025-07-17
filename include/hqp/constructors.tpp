#ifndef _HierarchicalQP_CONSTRUCTORS_TPP_
#define _HierarchicalQP_CONSTRUCTORS_TPP_

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS>
HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS>::HierarchicalQP(int m, int n)
  : row_{m}
  , col_{n}
  , primal_(n)
  , task_(n)
  , guess_(n)
  , force_(n)
  , tau_(n)
  , inverse_(n, n)
  , cholMetric_(n, n)
  , nullSpace_(n, n)
  , activeLowSet_(m)
  , activeUpSet_(m)
  , equalitySet_(m)
  , level_(m)
  , dual_(m)
  , lower_(m)
  , upper_(m)
  , vector_(m)
  , matrix_(m, n)
  , codLefts_(m, m) {
    guess_.setZero();
    cholMetric_.setIdentity();
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS>
template<int m, int n>
HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS>::HierarchicalQP(const Eigen::Matrix<double, m, n>& matrix)
  : HierarchicalQP(m, n) {
    matrix_ = matrix;
}

}  // namespace hqp

#endif  // _HierarchicalQP_CONSTRUCTORS_TPP_
