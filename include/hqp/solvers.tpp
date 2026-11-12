#ifndef HQP_SOLVERS_TPP
#define HQP_SOLVERS_TPP

#include <algorithm>
#include <limits>

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::solve() {
    if (equalitySet_.all()) {
        equality_hqp();
    } else {
        inequality_hqp();
    }

    guess_ = primal_;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::equality_hqp() {
    primal_ = guess_;
    k_ = std::numeric_limits<int>::max();
    increment_from(0);
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::inequality_hqp() {
    Eigen::Index idx;
    int row;
    double dual, mValue;
    bool isLowerBound;     // needed to distinguish between upper and lower bound in case they are both active

    int maxChanges = 500;  // Maximum constraint activations + deactivations
    changes        = 0;

    // primal_ solves the current active set, point_ moves towards it up to the first blocking constraint.
    equality_hqp();
    point_ = primal_;
    for (int h = 0; changes < maxChanges && h < lev_; ++h) {
        // Activate the constraints of level h already violated at point_.
        activate_violated(h, maxChanges);

        while (changes < maxChanges) {
            // Add tasks to the active set.
            double step = ratio_test(h, row, isLowerBound);
            if (row >= 0) {
                point_ += step * (primal_ - point_);
                decrement_from(level_(row));
                activate_constraint(row, isLowerBound);
                increment_from(level_(row));
                ++changes;
                continue;
            }
            point_ = primal_;

            // Remove tasks from the active set.
            dual_update(h);

            dual = -1;
            for (auto k = 0; k <= h; ++k) {
                int dim = breaksAct_(k) - breaksFix_(k);
                if (dim > 0) {
                    dual_.segment(breaksFix_(k), dim).noalias() =
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
                ++changes;
                continue;
            }

            for (int k = 0; k <= h; ++k) {
                for (int row = breaksFix_(k); row < breaksAct_(k); ++row) {
                    if (dual_(row) < -tolerance) {
                        lock_constraint(row);
                    }
                }
            }
            break;
        }
    }
    primal_ = point_;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::activate_violated(int h, int maxChanges) {
    int inactive = breaks_(h) - breaksAct_(h);
    if (inactive <= 0) {
        return;
    }

    vector_.segment(breaksAct_(h), inactive).noalias() = matrix_.middleRows(breaksAct_(h), inactive) * point_;
    bool activated = false;
    // activate_constraint only swaps rows already checked into i, so vector_ stays valid ahead of i.
    for (int i = breaksAct_(h), end = breaks_(h); i < end && changes < maxChanges; ++i) {
        bool low = lower_(i) - vector_(i) > tolerance;
        if (low || vector_(i) - upper_(i) > tolerance) {
            if (!activated) {
                decrement_from(h);
                activated = true;
            }
            activate_constraint(i, low);
            ++changes;
        }
    }
    if (activated) {
        increment_from(h);
    }
}

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
double HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::ratio_test(int h, int& row, bool& isLowerBound) {
    double step = 1;
    row         = -1;
    for (int k = 0; k <= h; ++k) {
        int dim = breaks_(k) - breaksAct_(k);
        if (dim > 0) {
            vector_.segment(breaksAct_(k), dim).noalias() = matrix_.middleRows(breaksAct_(k), dim) * primal_;
            for (int i = breaksAct_(k); i < breaks_(k); ++i) {
                bool up = vector_(i) - upper_(i) > tolerance;
                if (up || lower_(i) - vector_(i) > tolerance) {
                    // point_ satisfies the bounds, so the denominator is nonzero.
                    double from  = matrix_.row(i).dot(point_);
                    double bound = up ? upper_(i) : lower_(i);
                    double t     = std::clamp((bound - from) / (vector_(i) - from), 0.0, 1.0);
                    if (row < 0 || t < step) {
                        step         = t;
                        row          = i;
                        isLowerBound = !up;
                    }
                }
            }
        }
    }
    return step;
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::dual_update(int h) {
    int start = h == 0 ? 0 : breaks_(h - 1);
    int dim   = breaksAct_(h) - start;

    if (h >= k_) {
        dual_.segment(start, dim).noalias() =
          activeUpSet_.segment(start, dim).select(upper_.segment(start, dim), lower_.segment(start, dim)) -
          matrix_.middleRows(start, dim) * primal_;
    }
    tau_.noalias() = matrix_.middleRows(start, dim).transpose() * dual_.segment(start, dim);

    for (int k = h - 1; k >= 0; --k) {
        start = k == 0 ? 0 : breaks_(k - 1);
        dim   = breaksAct_(k) - start;
        if (dim > 0) {
            if (ranks_(k) && k < k_) {
                force_.head(ranks_(k)).noalias() = -inverse_.middleCols(col_ - dofs_(k), ranks_(k)).transpose() * tau_;
                codMids_[k]
                  .topLeftCorner(ranks_(k), ranks_(k))
                  .template triangularView<Eigen::Upper>()
                  .transpose()
                  .template solveInPlace<Eigen::OnTheLeft>(force_.head(ranks_(k)));
                dual_.segment(start, dim).noalias() =
                  codLefts_.block(start, 0, dim, ranks_(k)) * force_.head(ranks_(k));
                tau_.noalias() += matrix_.middleRows(start, dim).transpose() * dual_.segment(start, dim);
            } else {
                dual_.segment(start, dim).setZero();
            }
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::decrement_from(int level) {
    if (level >= k_) {
        return;
    }

    int start = level == 0 ? 0 : breaks_(level - 1);
    for (int k = level; k < lev_; ++k) {
        // if (k == k_) {k_ = parent;} no needed because it always calls increment_from right after
        if (breaksAct_(k) > start && ranks_(k) > 0) {
            primal_.noalias() -= inverse_.middleCols(col_ - dofs_(k), ranks_(k)) * task_.segment(col_ - dofs_(k), ranks_(k));
            dofs_(k) = ranks_(k) = 0;
            start                = breaks_(k);
        }
    }
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::increment_from(int level) {
    if (level >= k_) {
        return;
    }

    int parent = get_parent(level);
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


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::increment_primal(int parent, int k) {
    int dof = (parent < 0) ? col_ : dofs_(parent) - ranks_(parent);
    if (dof <= 0) {
        dofs_(k) = ranks_(k) = 0;
        return;
    }
    dofs_(k) = dof;

    int start  = k == 0 ? 0 : breaks_(k - 1);
    int n_rows = breaksAct_(k) - start;
    vector_.segment(start, n_rows).noalias() =
      activeUpSet_.segment(start, n_rows).select(upper_.segment(start, n_rows), lower_.segment(start, n_rows)) -
      matrix_.middleRows(start, n_rows) * primal_;

    // TODO: dynamically update tolerances to avoid tasks oscillations
    if (parent < 0) {
        nullSpace_.leftCols(dof).noalias() = cholMetric_.leftCols(dof);
    } else {
        nullSpace_.leftCols(dof).noalias() = codRights_[parent].middleCols(ranks_(parent), dof);
    }
    cod_.setThreshold(tolerance);
    cod_.compute(matrix_.middleRows(start, n_rows) * nullSpace_.leftCols(dof));
    ranks_(k)   = cod_.rank();
    int leftDof = dof - ranks_(k);

    if (leftDof > 0) {
        // In this case matrixZ() is not the identity, so Eigen computes it and is not garbage
        codRights_[k].leftCols(dof).noalias() =
          nullSpace_.leftCols(dof) * cod_.colsPermutation() * cod_.matrixZ().transpose();
    } else {
        codRights_[k].leftCols(dof).noalias() = nullSpace_.leftCols(dof) * cod_.colsPermutation();
    }
    auto codLeft = codLefts_.block(start, 0, n_rows, n_rows);
    codLeft.setIdentity();
    cod_.householderQ().applyThisOnTheLeft(codLeft);

    inverse_.middleCols(col_ - dof, ranks_(k)).noalias() = codRights_[k].leftCols(ranks_(k));
    task_.segment(col_ - dof, ranks_(k)).noalias() =
      codLefts_.block(start, 0, n_rows, ranks_(k)).transpose() * vector_.segment(start, n_rows);
    dual_.segment(start, n_rows).noalias() =
      vector_.segment(start, n_rows) -
      codLefts_.block(start, 0, n_rows, ranks_(k)) * task_.segment(col_ - dof, ranks_(k));
    cod_.matrixT()
      .topLeftCorner(ranks_(k), ranks_(k))
      .template triangularView<Eigen::Upper>()
      .template solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, ranks_(k)));
    primal_.noalias() += inverse_.middleCols(col_ - dof, ranks_(k)) * task_.segment(col_ - dof, ranks_(k));

    codMids_[k].topLeftCorner(ranks_(k), ranks_(k)).noalias() = cod_.matrixT().topLeftCorner(ranks_(k), ranks_(k));
}

}  // namespace hqp

#endif  // HQP_SOLVERS_TPP
