#ifndef _HierarchicalQP_ACCESSORS_TPP_
#define _HierarchicalQP_ACCESSORS_TPP_

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
int HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::get_parent(int level) {
    int parent = -1;
    for (int start = 0, k = 0; k < level; ++k) {
        parent = breaksAct_(k) > start ? k : parent;
        start  = breaks_(k);
    }
    return parent;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
Eigen::VectorXd HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::get_primal() {
    // TODO: move k = 0 in loop (leave int k out) and check style of all loops
    // TODO: move this logic in utils where both the stack and the solver are wrapped together in a new class
    // int k;
    // for (k = 0; k < k_ && sot[k].is_computed(); ++k) {}
    // if (k < k_ || k_ == 0) {
    solve();
    // }
    return primal_;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::set_metric(const Eigen::MatrixXd& metric) {
    assert(metric.rows() == metric.cols() && metric.rows() == col_ && "Metric must be a square matrix");
    Eigen::LLT<Eigen::MatrixXd> lltOf(metric);
    assert(metric.isApprox(metric.transpose()) && lltOf.info() != Eigen::NumericalIssue);
    cholMetric_.setIdentity();
    lltOf.matrixU().solveInPlace<Eigen::OnTheLeft>(cholMetric_);
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
template<typename MatrixType, typename LowerType, typename UpperType, typename BreaksType>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::set_problem(const MatrixType& matrix,
                                                                                const LowerType& lower,
                                                                                const UpperType& upper,
                                                                                const BreaksType& breaks) {
    // Compile-time checks for fixed-size matrices
    if constexpr (MatrixType::RowsAtCompileTime != Eigen::Dynamic && MatrixType::ColsAtCompileTime != Eigen::Dynamic) {
        static_assert(ROWS == MatrixType::RowsAtCompileTime || ROWS == Eigen::Dynamic,
                      "ROWS template parameter must match matrix rows or be Dynamic");
        static_assert(COLS == MatrixType::ColsAtCompileTime || COLS == Eigen::Dynamic,
                      "COLS template parameter must match matrix cols or be Dynamic");
    }
    if constexpr (BreaksType::RowsAtCompileTime != Eigen::Dynamic) {
        static_assert(MaxLevels >= BreaksType::RowsAtCompileTime || MaxLevels == -1,
                      "MaxLevels template parameter must be >= breaks size or -1");
    }

    assert(matrix.rows() == lower.size() && lower.size() == upper.size() &&
           "matrix, upper and lower must have the same number of rows");
    assert(breaks.size() > 0 && "breaks must not be empty");
    int prev = 0;
    for (int k = 0; k < breaks.size(); ++k) {
        assert(breaks(k) >= prev && "breaks must be increasing");
        prev = breaks(k);
    }
    assert(breaks(Eigen::last) == matrix.rows() && "The last break must be equal to matrix.rows()");
    assert((lower.array() <= upper.array()).all() && "Lower bounds must be less than or equal to upper bounds");

    matrix_ = matrix;
    lower_  = lower;
    upper_  = upper;
    breaks_ = breaks;

    equalitySet_  = lower.array() == upper.array();
    activeLowSet_ = equalitySet_.select(true, activeLowSet_);
    activeUpSet_  = equalitySet_.select(true, activeUpSet_);

    lev_ = breaks.size();
    // Resize vectors to the number of levels
    dofs_.resize(lev_);
    ranks_ = Eigen::VectorXi::Zero(lev_);
    codMids_.resize(lev_);
    codRights_.resize(lev_);
    breaksFix_.resize(lev_);
    breaksAct_.resize(lev_);

    for (int stop = 0, start = 0, k = 0; k < lev_; ++k) {
        int dim                    = breaks(k) - start;
        codMids_[k]                = nullSpace_;
        codRights_[k]              = nullSpace_;
        level_.segment(start, dim) = k * Eigen::VectorXi::Ones(dim);

        breaksFix_(k) = breaksAct_(k) = start;
        for (int row = start; row < breaks(k); ++row) {
            if (activeLowSet_(row)) {
                activate_constraint(row, true);
            } else if (activeUpSet_(row)) {
                activate_constraint(row, false);
            }
            if (equalitySet_(row)) {
                lock_constraint(breaksAct_(level_(row)) - 1);
            }
        }

        start = breaks(k);
    }
}

}  // namespace hqp

#endif  // _HierarchicalQP_ACCESSORS_TPP_
