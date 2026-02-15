#ifndef HQP_CONSTRUCTORS_TPP
#define HQP_CONSTRUCTORS_TPP

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::HierarchicalQP(int m, int n)
  : col_{n}
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
  , slackLow_(m)
  , slackUp_(m)
  , matrix_(m, n)
  , codLefts_(m, m)
  , perm_(m) {
    guess_.setZero();
    cholMetric_.setIdentity();
    activeLowSet_.setZero();
    activeUpSet_.setZero();
    for (int i = 0; i < m; ++i) perm_(i) = i;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
template<int m, int n, int l>
HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::HierarchicalQP(const Eigen::Matrix<double, m, n>& matrix,
                                                                              const Eigen::Vector<double, m>& lower,
                                                                              const Eigen::Vector<double, m>& upper,
                                                                              const Eigen::Vector<int, l>& breaks)
  : HierarchicalQP(m, n) {
    set_problem(matrix, lower, upper, breaks);
}

}  // namespace hqp

#endif  // HQP_CONSTRUCTORS_TPP
