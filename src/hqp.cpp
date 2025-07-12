#include <iostream>
#include "hqp.hpp"
#include "utils.hpp"

namespace hqp {

HierarchicalQP::HierarchicalQP(int m, int n)
  : row_{m}
  , col_{n}

  , primal_{Eigen::VectorXd::Zero(n)}
  , task_{Eigen::VectorXd::Zero(n)}
  , guess_{Eigen::VectorXd::Zero(n)}
  , force_{Eigen::VectorXd::Zero(n)}
  , inverse_{Eigen::MatrixXd::Zero(n, n)}
  , cholMetric_{Eigen::MatrixXd::Identity(n, n)}
  , nullSpace_{Eigen::MatrixXd::Identity(n, n)}

  , activeLowSet_{Eigen::VectorXi::Zero(m).cast<bool>()}
  , activeUpSet_{Eigen::VectorXi::Zero(m).cast<bool>()}
  , equalitySet_{Eigen::VectorXi::Zero(m).cast<bool>()}
  , lockedSet_{Eigen::VectorXi::Zero(m).cast<bool>()}
  , workSet_{Eigen::VectorXi::Zero(m).cast<bool>()}
  , level_{Eigen::VectorXi::Zero(m)}
  , dual_{Eigen::VectorXd::Zero(m)}
  , lower_{Eigen::VectorXd::Zero(m)}
  , upper_{Eigen::VectorXd::Zero(m)}
  , vector_{Eigen::VectorXd::Zero(m)}

  , matrix_{Eigen::MatrixXd::Zero(m, n)}
  , codLefts_{Eigen::MatrixXd::Zero(m, m)} {
}


void HierarchicalQP::solve() {
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
    Eigen::VectorXi rows = find(level_ >= k_ && !equalitySet_);
    activeLowSet_(rows).setZero();
    activeUpSet_(rows).setZero();
}


void HierarchicalQP::equality_hqp() {
    primal_.setZero();
    k_ = std::numeric_limits<int>::max();
    increment_from(0);
}


void HierarchicalQP::set_metric(const Eigen::MatrixXd& metric) {
    assert(metric.rows() == metric.cols() && metric.rows() == col_ && "Metric must be a square matrix");
    Eigen::LLT<Eigen::MatrixXd> lltOf(metric);
    assert(metric.isApprox(metric.transpose()) && lltOf.info() != Eigen::NumericalIssue);
    cholMetric_.setIdentity();
    lltOf.matrixU().solveInPlace<Eigen::OnTheLeft>(cholMetric_);
}


void HierarchicalQP::set_problem(Eigen::MatrixXd const& matrix,
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

    equalitySet_         = lower.array() == upper.array();
    Eigen::VectorXi rows = find(equalitySet_);
    activeLowSet_(rows) = activeUpSet_(rows) = true;

    int n_levels = breaks.size();
    // Resize vectors to the number of levels
    dofs_.resize(n_levels, -1);
    ranks_.resize(n_levels, 0);
    codMids_.resize(n_levels);
    codRights_.resize(n_levels);

    for (int start = 0, k = 0; k < n_levels; ++k) {
        int dim                    = breaks(k) - start;
        codMids_[k]                = Eigen::MatrixXd(dim, dim);
        codRights_[k]              = Eigen::MatrixXd(col_, col_);
        level_.segment(start, dim) = k * Eigen::VectorXi::Ones(dim);
        start                      = breaks(k);
    }
}


Eigen::VectorXd HierarchicalQP::get_primal() {
    // TODO: move k = 0 in loop (leave int k out) and check style of all loops
    // TODO: move this logic in utils where both the stack and the solver are wrapped together in a new class
    // int k;
    // for (k = 0; k < k_ && sot[k].is_computed(); ++k) {}
    // if (k < k_ || k_ == 0) {
    solve();
    // }
    return primal_;
}


void HierarchicalQP::inequality_hqp() {
    lockedSet_.setZero();
    Eigen::Index idx;
    int row;
    double slack, dual, mValue;
    bool isLowerBound;  // needed to distinguish between upper and lower bound in case they are both active

    equality_hqp();
    // TODO: replace maxIter with maxChanges for activations plus deactivations (each considered separately though)
    int maxIter = 500;
    for (auto iter = 0, h = 0; iter < maxIter && h <= level_.maxCoeff(); ++h) {
        slack = dual = 1;
        while ((slack > 0 || dual > 0) && iter < maxIter) {
            // Add tasks to the active set.
            slack = -1;
            for (auto k = 0; k <= level_.maxCoeff(); ++k) {
                if ((level_ == k && !activeUpSet_).any()) {
                    Eigen::VectorXi rows = find(level_ == k && !activeUpSet_);
                    mValue               = (matrix_(rows, Eigen::all) * primal_ - upper_(rows)).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = rows(idx);
                        isLowerBound = false;
                    }
                }

                if ((level_ == k && !activeLowSet_).any()) {
                    Eigen::VectorXi rows = find(level_ == k && !activeLowSet_);
                    mValue               = (lower_(rows) - matrix_(rows, Eigen::all) * primal_).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = rows(idx);
                        isLowerBound = true;
                    }
                }
            }
            if (slack > tolerance) {
                decrement_from(level_(row));
                if (isLowerBound) {
                    activeLowSet_(row) = true;
                } else {
                    activeUpSet_(row) = true;
                }
                increment_from(level_(row));
                continue;
            }

            // Remove tasks from the active set.
            dual_update(h);

            workSet_ = (activeLowSet_ || activeUpSet_) && !equalitySet_ && !lockedSet_;
            dual     = -1;
            for (auto k = 0; k <= h; ++k) {
                if ((level_ == k && workSet_).any()) {
                    // TODO: as it seems that both sides can be active, this might not be correct. Or rather it still
                    // considers one active constraint at a time?
                    Eigen::VectorXi rows = find(level_ == k && workSet_);
                    dual_(rows)          = activeUpSet_(rows).select(dual_(rows), -dual_(rows));
                    mValue               = dual_(rows).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > dual) {
                        dual = mValue;
                        row  = rows(idx);
                    }
                }
            }
            if (dual > tolerance) {
                decrement_from(level_(row));
                activeLowSet_(row) = false;
                activeUpSet_(row)  = false;
                increment_from(level_(row));
                continue;
            }

            if ((level_ <= h && workSet_).any()) {
                Eigen::VectorXi rows = find(level_ <= h && workSet_);
                lockedSet_(rows)     = dual_(rows).array() < -tolerance;
            }

            ++iter;
        }
    }
}


void HierarchicalQP::dual_update(int h) {
    Eigen::VectorXi rows = find(level_ == h && (activeLowSet_ || activeUpSet_));

    if (h >= k_) {
        dual_(rows) = activeUpSet_(rows).select(upper_(rows), lower_(rows)) - matrix_(rows, Eigen::all) * primal_;
        ranks_[h]   = 0;
    }
    tau_ = matrix_(rows, Eigen::all).transpose() * dual_(rows);

    for (auto dof = ranks_[h], k = h - 1; k >= 0; --k) {
        if ((level_ == k && (activeLowSet_ || activeUpSet_)).any()) {
            Eigen::VectorXi rows = find(level_ == k && (activeLowSet_ || activeUpSet_));
            if (ranks_[k] && k < k_) {
                dof                    += ranks_[k];
                force_.head(ranks_[k])  = -inverse_.middleCols(col_ - dof, ranks_[k]).transpose() * tau_;
                codMids_[k]
                  .topLeftCorner(ranks_[k], ranks_[k])
                  .triangularView<Eigen::Upper>()
                  .transpose()
                  .solveInPlace<Eigen::OnTheLeft>(force_.head(ranks_[k]));
                dual_(rows)  = codLefts_(rows, Eigen::seqN(0, ranks_[k])) * force_.head(ranks_[k]);
                tau_        += matrix_(rows, Eigen::all).transpose() * dual_(rows);
            } else {
                dual_(rows).setZero();
            }
        }
    }
}


// TODO: matrix and vector data accessed via aliases, e.g.: auto&& alias = vector_(rows);


void HierarchicalQP::decrement_from(int level) {
    if (level >= k_) {
        return;
    }

    for (int k = level; k <= level_.maxCoeff(); ++k) {
        // if (k == k_) {k_ = parent;} no needed because it always calls increment_from right after
        auto activeSet = level_ == k && (activeLowSet_ || activeUpSet_);
        if (activeSet.any() && ranks_[k] > 0) {
            primal_  -= inverse_.middleCols(col_ - dofs_[k], ranks_[k]) * task_.segment(col_ - dofs_[k], ranks_[k]);
            dofs_[k] = ranks_[k] = 0;
        }
    }
}


void HierarchicalQP::increment_from(int level) {
    if (level >= k_) {
        return;
    }

    int parent = -1;
    for (int h = 0; h < level; ++h) {
        parent = ranks_[h] > 0 ? h : parent;
    }

    int dof = (parent < 0) ? col_ : dofs_[parent] - ranks_[parent];
    for (k_ = level; dof > 0 && k_ <= level_.maxCoeff(); ++k_) {
        auto activeSet = level_ == k_ && (activeLowSet_ || activeUpSet_);
        if (activeSet.any()) {
            increment_primal(parent, k_);
            parent = k_;

            dof -= ranks_[k_];
        }
    }
}


void HierarchicalQP::increment_primal(int parent, int k) {
    int dof = (parent < 0) ? col_ : dofs_[parent] - ranks_[parent];
    if (dof <= 0) {
        dofs_[k] = ranks_[k] = 0;
        return;
    }
    dofs_[k] = dof;

    Eigen::VectorXi rows = find(level_ == k && (activeLowSet_ || activeUpSet_));
    int n_rows           = rows.size();
    vector_(rows)        = activeUpSet_(rows).select(upper_(rows), lower_(rows)) - matrix_(rows, Eigen::all) * primal_;

    // TODO: dynamically update tolerances to avoid tasks oscillations
    nullSpace_.leftCols(dof) = (parent < 0) ? cholMetric_ : codRights_[parent].middleCols(ranks_[parent], dof);
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(tolerance);
    cod.compute(matrix_(rows, Eigen::all) * nullSpace_.leftCols(dof));
    ranks_[k]   = cod.rank();
    int leftDof = dof - ranks_[k];

    codRights_[k].leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation();
    if (leftDof > 0) {
        // In this case matrixZ() is not the identity, so Eigen computes it and is not garbage
        codRights_[k].leftCols(dof) *= cod.matrixZ().transpose();
    }
    codLefts_(rows, Eigen::seqN(0, n_rows)) = cod.householderQ() * Eigen::MatrixXd::Identity(n_rows, n_rows);

    inverse_.middleCols(col_ - dof, ranks_[k]) = codRights_[k].leftCols(ranks_[k]);
    task_.segment(col_ - dof, ranks_[k])       = codLefts_(rows, Eigen::seqN(0, ranks_[k])).transpose() * vector_(rows);
    dual_(rows) = vector_(rows) - codLefts_(rows, Eigen::seqN(0, ranks_[k])) * task_.segment(col_ - dof, ranks_[k]);
    cod.matrixT()
      .topLeftCorner(ranks_[k], ranks_[k])
      .triangularView<Eigen::Upper>()
      .solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, ranks_[k]));
    primal_ += inverse_.middleCols(col_ - dof, ranks_[k]) * task_.segment(col_ - dof, ranks_[k]);

    codMids_[k].topLeftCorner(ranks_[k], ranks_[k]) = cod.matrixT().topLeftCorner(ranks_[k], ranks_[k]);
}


// TODO: upgrade to a logger keeping track of the active set
void HierarchicalQP::print_active_set() {
    std::cout << "Active set:\n";
    for (auto k = 0; k < k_; ++k) {
        if ((level_ == k && (activeLowSet_ || activeUpSet_)).any()) {
            std::cout << "\tLevel " << k << " -> constraints "
                      << find(level_ == k && (activeLowSet_ || activeUpSet_)).transpose() << "\n";
        }
    }
    std::cout << std::endl;
}

}  // namespace hqp
