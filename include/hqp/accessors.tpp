#ifndef _HierarchicalQP_ACCESSORS_TPP_
#define _HierarchicalQP_ACCESSORS_TPP_

#include <stdexcept>

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
double HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::get_level_cost(int k) {
    int start = k == 0 ? 0 : breaks_(k - 1);
    int dim   = breaks_(k) - start;
    vector_.segment(start, dim).noalias() = matrix_.middleRows(start, dim) * primal_;
    return (lower_.segment(start, dim) - vector_.segment(start, dim))
      .cwiseMax(vector_.segment(start, dim) - upper_.segment(start, dim))
      .cwiseMax(0.0)
      .squaredNorm();
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
Eigen::VectorXd HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::get_primal() {
    // TODO: move k = 0 in loop (leave int k out) and check style of all loops
    if (!primalValid_) {
        solve();
        primalValid_ = true;
    }
    return primal_;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
std::tuple<Eigen::Vector<double, ROWS>, Eigen::Vector<double, ROWS>>
  HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::get_slack() {
    if (!slacksValid_) {
        solve();
        Eigen::Vector<double, ROWS> vector = matrix_ * primal_;

        slackLow_ = vector - lower_;
        slackUp_  = vector - upper_;
        slackLow_ = (slackLow_.array() < 0).template cast<double>() * slackLow_.array();
        slackUp_  = (slackUp_.array() > 0).template cast<double>() * slackUp_.array();

        slacksValid_ = true;
    }
    return {slackLow_, slackUp_};
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::set_metric(const Eigen::MatrixXd& metric) {
    if (metric.rows() != metric.cols() || metric.rows() != col_) {
        throw std::invalid_argument("Metric must be a square matrix of size " + std::to_string(col_));
    }
    if (!metric.isApprox(metric.transpose(), tolerance)) {
        throw std::invalid_argument("Metric must be symmetric");
    }
    Eigen::LLT<Eigen::MatrixXd> lltOf(metric);
    if (lltOf.info() == Eigen::NumericalIssue) {
        throw std::invalid_argument("Metric must be positive definite");
    }
    cholMetric_.setIdentity();
    lltOf.matrixU().solveInPlace<Eigen::OnTheLeft>(cholMetric_);

    // Invalidate caches when metric changes
    primalValid_ = false;
    slacksValid_ = false;
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

    if (matrix.rows() != lower.size() || lower.size() != upper.size()) {
        throw std::invalid_argument("matrix, lower and upper must have the same number of rows");
    }
    if (breaks.size() <= 0) {
        throw std::invalid_argument("breaks must not be empty");
    }
    for (int prev = 0, k = 0; k < breaks.size(); ++k) {
        if (breaks(k) < prev) {
            throw std::invalid_argument("breaks must be non-decreasing");
        }
        prev = breaks(k);
    }
    if (breaks(Eigen::last) != matrix.rows()) {
        throw std::invalid_argument("The last break must equal matrix.rows()");
    }
    if (!(lower.array() <= upper.array()).all()) {
        throw std::invalid_argument("Lower bounds must be <= upper bounds");
    }

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

    for (int start = 0, k = 0; k < lev_; ++k) {
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

    primalValid_ = false;
    slacksValid_ = false;
}

}  // namespace hqp

#endif  // _HierarchicalQP_ACCESSORS_TPP_
