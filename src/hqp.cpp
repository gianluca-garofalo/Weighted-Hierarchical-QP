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
  , inverse_{Eigen::MatrixXd::Zero(n, n)}
  , cholMetric_{Eigen::MatrixXd::Identity(n, n)}

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
    int dof = col_;
    primal_.setZero();

    int lastActive = -1;
    for (k_ = 0; k_ <= level_.maxCoeff() && dof > 0; ++k_) {
        auto activeSet = level_ == k_ && (activeLowSet_ || activeUpSet_);
        if (activeSet.any()) {
            increment_primal(lastActive, k_);
            lastActive = k_;

            dof -= ranks_[k_];
        }
    }
}


void HierarchicalQP::set_metric(const Eigen::MatrixXd& metric) {
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
           "A, bu, bl must have the same number of rows");
    assert(breaks.size() > 0 && "break_points must not be empty");
    int prev = 0;
    for (auto k = 0; k < breaks.size(); ++k) {
        assert(breaks(k) >= prev && "break_points must be increasing");
        prev = breaks(k);
    }
    assert(breaks(Eigen::last) == matrix.rows() && "The last break_point must be equal to A.rows()");

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

    // TODO: replace maxIter with maxChanges for activations plus deactivations (each considered separately though)
    int maxIter = 500;
    for (auto iter = 0, h = 0; iter < maxIter && h <= level_.maxCoeff(); ++h) {
        slack = dual = 1;
        while ((slack > 0 || dual > 0) && iter < maxIter) {
            equality_hqp();

            // Add tasks to the active set.
            slack = -1;
            for (auto k = 0; k <= level_.maxCoeff(); ++k) {
                if ((level_ == k && !activeUpSet_).any()) {
                    Eigen::VectorXi rows   = find(level_ == k && !activeUpSet_);
                    Eigen::MatrixXd matrix = matrix_(rows, Eigen::all);
                    Eigen::VectorXd vector = upper_(rows);
                    mValue                 = (matrix * primal_ - vector).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = rows(idx);
                        isLowerBound = false;
                    }
                }

                if ((level_ == k && !activeLowSet_).any()) {
                    Eigen::VectorXi rows   = find(level_ == k && !activeLowSet_);
                    Eigen::MatrixXd matrix = matrix_(rows, Eigen::all);
                    Eigen::VectorXd vector = lower_(rows);
                    mValue                 = (vector - matrix * primal_).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = rows(idx);
                        isLowerBound = true;
                    }
                }
            }
            if (slack > tolerance) {
                if (isLowerBound) {
                    activeLowSet_(row) = true;
                } else {
                    activeUpSet_(row) = true;
                }
                continue;
            }

            // Remove tasks from the active set.
            dual_update(h);

            workSet_ = (activeLowSet_ || activeUpSet_) && !equalitySet_ && !lockedSet_;
            dual     = -1;
            for (auto k = 0; k <= h; ++k) {
                if ((level_ == k && workSet_).any()) {
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
                activeLowSet_(row) = false;
                activeUpSet_(row)  = false;
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
    Eigen::VectorXi rows   = find(level_ == h && (activeLowSet_ || activeUpSet_));
    Eigen::MatrixXd matrix = matrix_(rows, Eigen::all);
    Eigen::VectorXd vector = activeUpSet_(rows).select(upper_(rows), lower_(rows));

    if (h >= k_) {
        dual_(rows) = vector - matrix * primal_;
        ranks_[h]   = 0;
    }
    Eigen::VectorXd tau = matrix.transpose() * dual_(rows);
    Eigen::VectorXd f   = primal_;

    for (auto dof = ranks_[h], k = h - 1; k >= 0; --k) {
        if ((level_ == k && (activeLowSet_ || activeUpSet_)).any()) {
            Eigen::VectorXi rows = find(level_ == k && (activeLowSet_ || activeUpSet_));
            if (ranks_[k] && k < k_) {
                dof               += ranks_[k];
                f.head(ranks_[k])  = -inverse_.middleCols(col_ - dof, ranks_[k]).transpose() * tau;
                codMids_[k]
                  .topLeftCorner(ranks_[k], ranks_[k])
                  .triangularView<Eigen::Upper>()
                  .transpose()
                  .solveInPlace<Eigen::OnTheLeft>(f.head(ranks_[k]));
                dual_(rows)             = codLefts_(rows, Eigen::seqN(0, ranks_[k])) * f.head(ranks_[k]);
                Eigen::MatrixXd matrix  = matrix_(rows, Eigen::all);
                tau                    += matrix.transpose() * dual_(rows);
            } else {
                dual_(rows).setZero();
            }
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

    Eigen::VectorXi rows    = find(level_ == k && (activeLowSet_ || activeUpSet_));
    Eigen::MatrixXd matrix  = matrix_(rows, Eigen::all);
    Eigen::VectorXd vector  = activeUpSet_(rows).select(upper_(rows), lower_(rows));
    vector                 -= matrix * primal_;

    // sot[k].parent_(rows) = parent;

    Eigen::MatrixXd nullSpace = (parent < 0) ? cholMetric_ : codRights_[parent].middleCols(ranks_[parent], dof);
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(tolerance);
    cod.compute(matrix * nullSpace);
    int rank    = cod.rank();
    int leftDof = dof - rank;

    codRights_[k].leftCols(dof) = nullSpace * cod.colsPermutation();
    if (leftDof > 0) {
        // In this case matrixZ() is not the identity, so Eigen computes it and is not garbage
        codRights_[k].leftCols(dof) *= cod.matrixZ().transpose();
    }
    Eigen::MatrixXd codLeft = cod.householderQ();

    inverse_.middleCols(col_ - dof, rank) = codRights_[k].leftCols(rank);
    task_.segment(col_ - dof, rank)       = codLeft.leftCols(rank).transpose() * vector;
    dual_(rows)                           = vector - codLeft.leftCols(rank) * task_.segment(col_ - dof, rank);
    cod.matrixT()
      .topLeftCorner(rank, rank)
      .triangularView<Eigen::Upper>()
      .solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, rank));
    primal_ += inverse_.middleCols(col_ - dof, rank) * task_.segment(col_ - dof, rank);

    ranks_[k]                             = rank;
    codMids_[k].topLeftCorner(rank, rank) = cod.matrixT().topLeftCorner(rank, rank);
    codLefts_(rows, Eigen::seqN(0, rank)) = codLeft.leftCols(rank);
}


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
