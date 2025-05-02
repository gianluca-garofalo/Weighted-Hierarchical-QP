#include <iostream>
#include "hqp.hpp"
#include "utils.hpp"

namespace hqp {

HierarchicalQP::HierarchicalQP(unsigned int n)
  : col_{n}
  , primal_{Eigen::VectorXd::Zero(n)}
  , task_{Eigen::VectorXd::Zero(n)}
  , guess_{Eigen::VectorXd::Zero(n)}
  , inverse_{Eigen::MatrixXd::Zero(n, n)}
  , cholMetric_{Eigen::MatrixXd::Identity(n, n)} {
}


void HierarchicalQP::solve() {
    bool isAllEquality = true;
    for (auto& task : sot) {
        task->codRight_.resize(col_, col_);
        if (!task->activeSet_.size()) {
            task->activeSet_ = task->equalitySet_;
        }
        isAllEquality = isAllEquality && task->equalitySet_.all();
    }
    if (isAllEquality) {
        equality_hqp();
    } else {
        inequality_hqp();
    }

    // Shift problem back
    primal_ += guess_;
    guess_   = primal_;

    // Deactivate unused tasks for next guess
    for (unsigned int k = k_; k < sot.size(); ++k) {
        auto rows = find(!sot[k]->equalitySet_);
        sot[k]->activeSet_(rows).setZero();
    }
}


void HierarchicalQP::equality_hqp() {
    primal_.setZero();
    auto dof = col_;
    k_       = 0;
    Eigen::MatrixXd nullSpace{cholMetric_};
    while (k_ < sot.size() && dof > 0) {
        if (sot[k_]->activeSet_.any()) {
            auto rows              = find(sot[k_]->activeSet_);
            auto [matrix, vector]  = get_task(sot[k_], rows);
            vector                -= matrix * primal_;

            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
            // TODO: dynamically update tolerances to avoid tasks oscillations
            cod.setThreshold(sot[k_]->tolerance);
            cod.compute(matrix * nullSpace.leftCols(dof));
            auto rank    = cod.rank();
            auto leftDof = dof - rank;
            if (leftDof > 0) {
                sot[k_]->codRight_.leftCols(dof) = nullSpace.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
                nullSpace.leftCols(leftDof) = sot[k_]->codRight_.middleCols(rank, leftDof);
            } else {
                // In this case matrixZ() is the identity, so Eigen does not compute it and matrixZ() returns garbage
                sot[k_]->codRight_.leftCols(dof) = nullSpace.leftCols(dof) * cod.colsPermutation();
            }
            Eigen::MatrixXd codLeft = cod.householderQ();

            inverse_.middleCols(col_ - dof, rank) = sot[k_]->codRight_.leftCols(rank);
            task_.segment(col_ - dof, rank)       = codLeft.leftCols(rank).transpose() * vector;
            sot[k_]->slack_(rows)                 = codLeft.leftCols(rank) * task_.segment(col_ - dof, rank) - vector;
            cod.matrixT()
              .topLeftCorner(rank, rank)
              .triangularView<Eigen::Upper>()
              .solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, rank));
            primal_ += inverse_.middleCols(col_ - dof, rank) * task_.segment(col_ - dof, rank);

            dof                                           = leftDof;
            sot[k_]->rank_                                = rank;
            sot[k_]->codMid_.topLeftCorner(rank, rank)    = cod.matrixT().topLeftCorner(rank, rank);
            sot[k_]->codLeft_(rows, Eigen::seqN(0, rank)) = codLeft.leftCols(rank);
        }
        k_++;
    }
}


void HierarchicalQP::set_metric(const Eigen::MatrixXd& metric) {
    Eigen::LLT<Eigen::MatrixXd> lltOf(metric);
    assert(metric.isApprox(metric.transpose()) && lltOf.info() != Eigen::NumericalIssue);
    cholMetric_.setIdentity();
    lltOf.matrixU().solveInPlace<Eigen::OnTheLeft>(cholMetric_);
}


Eigen::VectorXd HierarchicalQP::get_primal() {
    unsigned int k = 0;
    while (k < k_ && sot[k]->is_computed()) {
        k++;
    }
    if (k < k_ || k_ == 0) {
        solve();
    }
    return primal_;
}


void HierarchicalQP::inequality_hqp() {
    for (auto& task : sot) {
        task->lockedSet_.setZero();
        task->dual_.setZero();
    }
    unsigned int h      = 0;
    bool isActiveSetNew = false;

    while (h < sot.size()) {
        equality_hqp();
        for (unsigned int k = 0; k < k_ && !isActiveSetNew; ) {
            auto rows                = find(!sot[k]->activeSet_);
            auto [matrix, vector]    = get_task(sot[k], rows);
            sot[k]->activeSet_(rows) = (vector - matrix * primal_).array() > tolerance;
            isActiveSetNew           = sot[k]->activeSet_(rows).any();
            
            if (isActiveSetNew) {
                auto tmp_rows = find(sot[k]->activeSet_(rows));
                update_active_set(k, rows(tmp_rows(0)));
                k = 0;
            } else {
                k++;
            }
        }

        // Remove tasks from the active set.
        auto rows             = find(sot[h]->activeSet_);
        auto [matrix, vector] = get_task(sot[h], rows);

        if (h >= k_) {
            sot[h]->slack_(rows) = matrix * primal_ - vector;
        }
        sot[h]->dual_(rows) = sot[h]->slack_(rows);
        dual_update(h, matrix.transpose() * sot[h]->dual_(rows));

        for (unsigned int k = 0; k <= h && !isActiveSetNew; ++k) {
            sot[k]->workSet_ = sot[k]->activeSet_ && !sot[k]->equalitySet_ && !sot[k]->lockedSet_;
            if (sot[k]->workSet_.any()) {
                auto rows                = find(sot[k]->workSet_);
                auto test                = (sot[k]->dual_(rows)).array() > tolerance;
                sot[k]->activeSet_(rows) = !test;
                sot[k]->lockedSet_(rows) = (sot[k]->dual_(rows)).array() < 0;
                isActiveSetNew           = k < k_ && test.any();
            }
        }

        if (!isActiveSetNew) {
            h++;
        }
    }
}


void HierarchicalQP::dual_update(unsigned int h, const Eigen::VectorXd& tau) {
    auto dof = col_;
    for (unsigned int k = 0; k < h; ++k) {
        if (dof > 0 && sot[k]->activeSet_.any()) {
            auto leftDof = dof - sot[k]->rank_;
            auto rows    = find(sot[k]->activeSet_);
            if (rows.any()) {
                Eigen::VectorXd f = inverse_.middleCols(col_ - dof, sot[k]->rank_).transpose() * tau;
                sot[k]
                  ->codMid_.topLeftCorner(sot[k]->rank_, sot[k]->rank_)
                  .triangularView<Eigen::Upper>()
                  .transpose()
                  .solveInPlace<Eigen::OnTheLeft>(f);
                sot[k]->dual_(rows) = -sot[k]->codLeft_(rows, Eigen::seqN(0, sot[k]->rank_)) * f;
            }

            dof = leftDof;
        } else {
            sot[k]->dual_.setZero();
        }
    }
}


std::tuple<Eigen::MatrixXd, Eigen::VectorXd> HierarchicalQP::get_task(TaskPtr task, const Eigen::VectorXi& rows) {
    if (!task->is_computed()) {
        task->compute();
        assert(task->indices_.maxCoeff() < col_);
        if (task->weight_.size()) {
            // Weight subtasks within task
            task->matrix_ = task->weight_.matrixU() * task->matrix_;
            task->vector_ = task->weight_.matrixU() * task->vector_;
        }
        // Shift problem to the origin
        task->vector_ -= task->matrix_ * guess_(task->indices_);
    }
    Eigen::MatrixXd A             = Eigen::MatrixXd::Zero(rows.size(), col_);
    A(Eigen::all, task->indices_) = task->matrix_(rows, Eigen::all);
    return {A, task->vector_(rows)};
}


bool HierarchicalQP::update_active_set(unsigned int level, unsigned int row) {
    auto dof = col_;
    Eigen::MatrixXd codRight{Eigen::MatrixXd::Zero(col_, col_)};
    Eigen::MatrixXd nullSpace{cholMetric_};

    bool hasConflict = false;
    for (unsigned int k = 0; !hasConflict && dof > 0 && k < sot.size(); ++k) {
        if (sot[k]->activeSet_.any()) {
            if (k >= level) {
                Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
                cod.setThreshold(sot[k]->tolerance);
                cod.compute(sot[level]->matrix_.row(row) * nullSpace.leftCols(dof));
                if (cod.rank() == 0) {
                    return false;
                }

                if (dof > 1) {
                    codRight.leftCols(dof) = nullSpace.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
                } else {
                    // In this case matrixZ() is the identity, so Eigen does not compute it and matrixZ() returns garbage
                    codRight.leftCols(dof) = nullSpace.leftCols(dof) * cod.colsPermutation();
                }
    
                Eigen::VectorXi idx_active = find(sot[k]->activeSet_&& !sot[k]->equalitySet_);
                Eigen::MatrixXd matrix = sot[k]->matrix_(idx_active, Eigen::all) * codRight.middleCols(1, dof - 1);
                Eigen::VectorXi conflict_row = idx_active(find(matrix.rowwise().any().array() == false));
                hasConflict = conflict_row.size() >= 1;
                if (hasConflict) {
                    sot[k]->matrix_.row(conflict_row(0)).swap(sot[level]->matrix_.row(row));
                    sot[k]->vector_.row(conflict_row(0)).swap(sot[level]->vector_.row(row));
                    // levels_[k].values_(conflict_row(0)).swap(levels_[level]->values_(row));
                }
            }

            auto leftDof = dof - sot[k]->rank_;
            if (leftDof > 0) {
                nullSpace.leftCols(leftDof) = sot[k]->codRight_.middleCols(sot[k]->rank_, leftDof);
            }
            dof = leftDof;
        }
    }

    return hasConflict;
}


// TODO: upgrade to a logger keeping track of the active set
void HierarchicalQP::print_active_set() {
    std::cout << "Active set:\n";
    for (unsigned int k = 0; const auto& task : sot) {
        if (k < k_ && task->activeSet_.any()) {
            std::cout << "\tLevel " << k << " -> constraints " << find(task->activeSet_).transpose() << "\n";
        }
        k++;
    }
    std::cout << std::endl;
}

}  // namespace hqp
