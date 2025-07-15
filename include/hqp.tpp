#include <iostream>

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels>
HierarchicalQP<MaxRows, MaxCols, MaxLevels>::HierarchicalQP(int m, int n)
  : row_{m}
  , col_{n} {
    primal_.resize(n);
    task_.resize(n);
    guess_.resize(n);
    force_.resize(n);
    tau_.resize(n);
    inverse_.resize(n, n);
    cholMetric_.resize(n, n);
    nullSpace_.resize(n, n);
    activeLowSet_.resize(m);
    activeUpSet_.resize(m);
    equalitySet_.resize(m);
    level_.resize(m);
    dual_.resize(m);
    lower_.resize(m);
    upper_.resize(m);
    vector_.resize(m);
    matrix_.resize(m, n);
    codLefts_.resize(m, m);

    guess_.setZero();
    cholMetric_.setIdentity();
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::solve() {
    // Shift problem to the origin
    lower_ -= matrix_ * guess_;
    upper_ -= matrix_ * guess_;

    if (equalitySet_.all()) {
        equality_hqp();
    } else {
        inequality_hqp();
    }

    // Shift problem back
    primal_ += guess_;
    guess_   = primal_;

    // Deactivate unused tasks for next guess
    for (int level = k_; level < lev_; ++level) {
        for (int row = level == 0 ? 0 : breaks_(level - 1); row < breaksAct_(level); ++row) {
            if (!equalitySet_(row)) {
                deactivate_constraint(row);
            }
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::equality_hqp() {
    primal_.setZero();
    k_ = std::numeric_limits<int>::max();
    increment_from(0);
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::set_metric(const Eigen::MatrixXd& metric) {
    assert(metric.rows() == metric.cols() && metric.rows() == col_ && "Metric must be a square matrix");
    Eigen::LLT<Eigen::MatrixXd> lltOf(metric);
    assert(metric.isApprox(metric.transpose()) && lltOf.info() != Eigen::NumericalIssue);
    cholMetric_.setIdentity();
    lltOf.matrixU().solveInPlace<Eigen::OnTheLeft>(cholMetric_);
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::set_problem(Eigen::MatrixXd const& matrix,
                                                              Eigen::VectorXd const& lower,
                                                              Eigen::VectorXd const& upper,
                                                              Eigen::VectorXi const& breaks) {
    assert(matrix.rows() == lower.size() && lower.size() == upper.size() &&
           "matrix, upper and lower must have the same number of rows");
    assert(breaks.size() > 0 && "breaks must not be empty");
    int prev = 0;
    for (int k = 0; k < breaks.size(); ++k) {
        assert(breaks(k) >= prev && "breaks must be increasing");
        prev = breaks(k);
    }
    assert(breaks(Eigen::last) == matrix.rows() && "The last break must be equal to matrix.rows()");

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
        codMids_[k]                = Eigen::MatrixXd(dim, dim);
        codRights_[k]              = Eigen::MatrixXd(col_, col_);
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


template<int MaxRows, int MaxCols, int MaxLevels>
Eigen::VectorXd HierarchicalQP<MaxRows, MaxCols, MaxLevels>::get_primal() {
    // TODO: move k = 0 in loop (leave int k out) and check style of all loops
    // TODO: move this logic in utils where both the stack and the solver are wrapped together in a new class
    // int k;
    // for (k = 0; k < k_ && sot[k].is_computed(); ++k) {}
    // if (k < k_ || k_ == 0) {
    solve();
    // }
    return primal_;
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::inequality_hqp() {
    Eigen::Index idx;
    int row;
    double slack, dual, mValue;
    bool isLowerBound;  // needed to distinguish between upper and lower bound in case they are both active

    equality_hqp();
    // TODO: replace maxIter with maxChanges for activations plus deactivations (each considered separately though)
    int maxIter = 500;
    for (auto iter = 0, h = 0; iter < maxIter && h < lev_; ++h) {
        slack = dual = 1;
        while ((slack > 0 || dual > 0) && iter < maxIter) {
            // Add tasks to the active set.
            slack = -1;
            for (int k = 0; k < lev_; ++k) {
                int dim = breaks_(k) - breaksAct_(k);
                if (dim > 0) {
                    vector_.segment(breaksAct_(k), dim) =
                      (!activeUpSet_.segment(breaksAct_(k), dim)).select(upper_.segment(breaksAct_(k), dim), 1e9);
                    mValue = (matrix_.middleRows(breaksAct_(k), dim) * primal_ - vector_.segment(breaksAct_(k), dim))
                               .maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = breaksAct_(k) + idx;
                        isLowerBound = false;
                    }

                    vector_.segment(breaksAct_(k), dim) =
                      (!activeLowSet_.segment(breaksAct_(k), dim)).select(lower_.segment(breaksAct_(k), dim), -1e9);
                    mValue = (vector_.segment(breaksAct_(k), dim) - matrix_.middleRows(breaksAct_(k), dim) * primal_)
                               .maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = breaksAct_(k) + idx;
                        isLowerBound = true;
                    }
                }
            }
            if (slack > tolerance) {
                decrement_from(level_(row));
                activate_constraint(row, isLowerBound);
                increment_from(level_(row));
                continue;
            }

            // Remove tasks from the active set.
            dual_update(h);

            dual = -1;
            for (auto k = 0; k <= h; ++k) {
                int dim = breaksAct_(k) - breaksFix_(k);
                if (dim > 0) {
                    // TODO: as it seems that both sides can be active, this might not be correct. Or rather it still
                    // considers one active constraint at a time?
                    dual_.segment(breaksFix_(k), dim) =
                      activeUpSet_.segment(breaksFix_(k), dim)
                        .select(dual_.segment(breaksFix_(k), dim), -dual_.segment(breaksFix_(k), dim));
                    mValue = dual_.segment(breaksFix_(k), dim).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > dual) {
                        dual = mValue;
                        row  = breaksFix_(k) + idx;
                    }
                }
            }
            if (dual > tolerance) {
                decrement_from(level_(row));
                deactivate_constraint(row);
                increment_from(level_(row));
                continue;
            }

            for (int k = 0; k <= h; ++k) {
                for (int row = breaksFix_(k); row < breaksAct_(k); ++row) {
                    if (dual_(row) < -tolerance) {
                        lock_constraint(row);
                    }
                }
            }

            ++iter;
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::dual_update(int h) {
    int start = h == 0 ? 0 : breaks_(h - 1);
    int dim   = breaksAct_(h) - start;

    if (h >= k_) {
        dual_.segment(start, dim) =
          activeUpSet_.segment(start, dim).select(upper_.segment(start, dim), lower_.segment(start, dim)) -
          matrix_.middleRows(start, dim) * primal_;
    }
    tau_ = matrix_.middleRows(start, dim).transpose() * dual_.segment(start, dim);

    for (int dof = ranks_(h), k = h - 1; k >= 0; --k) {
        start = k == 0 ? 0 : breaks_(k - 1);
        dim   = breaksAct_(k) - start;
        if (dim > 0) {
            if (ranks_(k) && k < k_) {
                dof                    += ranks_(k);
                force_.head(ranks_(k))  = -inverse_.middleCols(col_ - dof, ranks_(k)).transpose() * tau_;
                codMids_[k]
                  .topLeftCorner(ranks_(k), ranks_(k))
                  .template triangularView<Eigen::Upper>()
                  .transpose()
                  .template solveInPlace<Eigen::OnTheLeft>(force_.head(ranks_(k)));
                dual_.segment(start, dim)  = codLefts_.block(start, 0, dim, ranks_(k)) * force_.head(ranks_(k));
                tau_                      += matrix_.middleRows(start, dim).transpose() * dual_.segment(start, dim);
            } else {
                dual_.segment(start, dim).setZero();
            }
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::decrement_from(int level) {
    if (level >= k_) {
        return;
    }

    int start = level == 0 ? 0 : breaks_(level - 1);
    for (int k = level; k < lev_; ++k) {
        // if (k == k_) {k_ = parent;} no needed because it always calls increment_from right after
        if (breaksAct_(k) > start && ranks_(k) > 0) {
            primal_  -= inverse_.middleCols(col_ - dofs_(k), ranks_(k)) * task_.segment(col_ - dofs_(k), ranks_(k));
            dofs_(k) = ranks_(k) = 0;
            start                = breaks_(k);
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::increment_from(int level) {
    if (level >= k_) {
        return;
    }

    int parent = -1;
    for (int h = 0; h < level; ++h) {
        parent = ranks_[h] > 0 ? h : parent;
    }

    int dof    = (parent < 0) ? col_ : dofs_(parent) - ranks_(parent);
    int start  = level == 0 ? 0 : breaks_(level - 1);
    for (k_ = level; dof > 0 && k_ < lev_; ++k_) {
        if (breaksAct_(k_) > start) {
            increment_primal(parent, k_);
            parent = k_;

            dof   -= ranks_(k_);
            start  = breaks_(k_);
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::increment_primal(int parent, int k) {
    int dof = (parent < 0) ? col_ : dofs_(parent) - ranks_(parent);
    if (dof <= 0) {
        dofs_(k) = ranks_(k) = 0;
        return;
    }
    dofs_(k) = dof;

    int start  = k == 0 ? 0 : breaks_(k - 1);
    int n_rows = breaksAct_(k) - start;
    vector_.segment(start, n_rows) =
      activeUpSet_.segment(start, n_rows).select(upper_.segment(start, n_rows), lower_.segment(start, n_rows)) -
      matrix_.middleRows(start, n_rows) * primal_;

    // TODO: dynamically update tolerances to avoid tasks oscillations
    nullSpace_.leftCols(dof) = (parent < 0) ? cholMetric_ : codRights_[parent].middleCols(ranks_(parent), dof);
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(tolerance);
    cod.compute(matrix_.middleRows(start, n_rows) * nullSpace_.leftCols(dof));
    ranks_(k)   = cod.rank();
    int leftDof = dof - ranks_(k);

    codRights_[k].leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation();
    if (leftDof > 0) {
        // In this case matrixZ() is not the identity, so Eigen computes it and is not garbage
        codRights_[k].leftCols(dof) *= cod.matrixZ().transpose();
    }
    codLefts_.block(start, 0, n_rows, n_rows) = cod.householderQ() * Eigen::MatrixXd::Identity(n_rows, n_rows);

    inverse_.middleCols(col_ - dof, ranks_(k)) = codRights_[k].leftCols(ranks_(k));
    task_.segment(col_ - dof, ranks_(k)) =
      codLefts_.block(start, 0, n_rows, ranks_(k)).transpose() * vector_.segment(start, n_rows);
    dual_.segment(start, n_rows) = vector_.segment(start, n_rows) -
                                   codLefts_.block(start, 0, n_rows, ranks_(k)) * task_.segment(col_ - dof, ranks_(k));
    cod.matrixT()
      .topLeftCorner(ranks_(k), ranks_(k))
      .template triangularView<Eigen::Upper>()
      .template solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, ranks_(k)));
    primal_ += inverse_.middleCols(col_ - dof, ranks_(k)) * task_.segment(col_ - dof, ranks_(k));

    codMids_[k].topLeftCorner(ranks_(k), ranks_(k)) = cod.matrixT().topLeftCorner(ranks_(k), ranks_(k));
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::lock_constraint(int row) {
    if (breaksFix_(level_(row)) < breaksAct_(level_(row))) {
        swap_constraints(breaksFix_(level_(row)), row);
    } else {
        throw std::runtime_error("Cannot lock more constraints than the active ones.");
    }
    ++breaksFix_(level_(row));
}

template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::activate_constraint(int row, bool isLowerBound) {
    if (isLowerBound) {
        activeLowSet_(row) = true;
    } else {
        activeUpSet_(row) = true;
    }

    if (breaksAct_(level_(row)) < breaks_(level_(row))) {
        swap_constraints(breaksAct_(level_(row)), row);
    } else {
        throw std::runtime_error("Cannot activate more constraints than the available ones.");
    }
    ++breaksAct_(level_(row));
}

template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::deactivate_constraint(int row) {
    activeLowSet_(row) = false;
    activeUpSet_(row)  = false;

    swap_constraints(--breaksAct_(level_(row)), row);
}


template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::swap_constraints(int i, int j) {
    if (i == j) {
        return;
    }

    std::swap(activeLowSet_(i), activeLowSet_(j));
    std::swap(activeUpSet_(i), activeUpSet_(j));
    std::swap(equalitySet_(i), equalitySet_(j));
    std::swap(lower_(i), lower_(j));
    std::swap(upper_(i), upper_(j));
    std::swap(dual_(i), dual_(j));
    matrix_.row(i).swap(matrix_.row(j));
    codLefts_.row(i).swap(codLefts_.row(j));
}


// TODO: upgrade to a logger keeping track of the active set
template<int MaxRows, int MaxCols, int MaxLevels>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels>::print_active_set() {
    std::cout << "Active set:\n";
    for (int start = 0, k = 0; k < k_; ++k) {
        std::cout << "\tLevel " << k << ":\n";
        for (int row = start; row < breaksAct_(k); ++row) {
            std::cout << "\t\t" << lower_(row) << " < " << matrix_.row(row) << " < " << upper_(row) << "\n";
        }
        start = breaks_(k);
    }
    std::cout << std::endl;
}

}  // namespace hqp
