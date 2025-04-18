#include <iostream>
#include "hqp.hpp"
#include "utils.hpp"

namespace hqp {

HierarchicalQP::HierarchicalQP(uint n)
  : col_{n}
  , primal_{Eigen::VectorXd::Zero(n)}
  , task_{Eigen::VectorXd::Zero(n)}
  , guess_{Eigen::VectorXd::Zero(n)}
  , inverse_{Eigen::MatrixXd::Zero(n, n)}
  , nullSpace_{Eigen::MatrixXd::Identity(n, n)}
  , codRight_{Eigen::MatrixXd::Zero(n, n)}
  , cholMetric_{Eigen::MatrixXd::Identity(n, n)} {
#if DEBUG
    logger.log("HierarchicalQP initialized with " + std::to_string(n) + " degrees of freedom.");
#endif
}


void HierarchicalQP::solve() {
    bool isAllEquality = true;
    for (auto& task : sot) {
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
    for (uint k = k_; k < sot.size(); ++k) {
        auto rows = find(!sot[k]->equalitySet_);
        sot[k]->activeSet_(rows).setZero();
    }
#if DEBUG
    logger.log(print_active_set());
#endif
}


void HierarchicalQP::equality_hqp() {
    primal_.setZero();
    auto dof   = col_;
    k_         = 0;
    nullSpace_ = cholMetric_;
    while (k_ < sot.size() && dof > 0) {
        if (sot[k_]->activeSet_.any()) {
            auto rows              = find(sot[k_]->activeSet_);
            auto [matrix, vector]  = get_task(sot[k_], rows);
            vector                -= matrix * primal_;

            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
            // TODO: dynamically update tolerances to avoid tasks oscillations
            cod.setThreshold(sot[k_]->tolerance);
            cod.compute(matrix * nullSpace_.leftCols(dof));
            auto rank    = cod.rank();
            auto leftDof = dof - rank;
            if (leftDof > 0) {
                codRight_.leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
                nullSpace_.leftCols(leftDof) = codRight_.rightCols(leftDof);
            } else {
                // In this case matrixZ() is the identity, so Eigen does not compute it and matrixZ() returns garbage
                codRight_.leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation();
            }
            Eigen::MatrixXd codLeft_ = cod.householderQ();

            inverse_.middleCols(col_ - dof, rank) = codRight_.leftCols(rank);
            task_.segment(col_ - dof, rank)       = codLeft_.leftCols(rank).transpose() * vector;
            sot[k_]->slack_(rows)                 = codLeft_.leftCols(rank) * task_.segment(col_ - dof, rank) - vector;
            cod.matrixT()
              .topLeftCorner(rank, rank)
              .triangularView<Eigen::Upper>()
              .solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, rank));
            primal_ += inverse_.middleCols(col_ - dof, rank) * task_.segment(col_ - dof, rank);

            dof                                           = leftDof;
            sot[k_]->rank_                                = rank;
            sot[k_]->codMid_.topLeftCorner(rank, rank)    = cod.matrixT().topLeftCorner(rank, rank);
            sot[k_]->codLeft_(rows, Eigen::seqN(0, rank)) = codLeft_.leftCols(rank);
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
    uint k = 0;
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
    uint h              = 0;
    bool isActiveSetNew = true;

    while (h < sot.size()) {
        while (isActiveSetNew) {
            equality_hqp();

            // Add tasks to the active set.
            isActiveSetNew = false;
            for (uint k = 0; k < k_ && !isActiveSetNew; ++k) {
                auto rows                = find(!sot[k]->activeSet_);
                auto [matrix, vector]    = get_task(sot[k], rows);
                sot[k]->activeSet_(rows) = (vector - matrix * primal_).array() > tolerance;
                isActiveSetNew           = sot[k]->activeSet_(rows).any();
            }
        }

        // Remove tasks from the active set.
        for (uint k = 0; k <= h; ++k) {
            sot[k]->workSet_ = sot[k]->activeSet_ && !sot[k]->equalitySet_ && !sot[k]->lockedSet_;
        }

        if (sot[h]->workSet_.any()) {
            auto rows             = find(sot[h]->workSet_);
            auto [matrix, vector] = get_task(sot[h], rows);

            if (h >= k_) {
                sot[h]->slack_(rows) = matrix * primal_ - vector;
            }
            sot[h]->dual_(rows) = sot[h]->slack_(rows);
            dual_update(h, matrix.transpose() * sot[h]->dual_(rows));

            for (uint k = 0; k <= h && !isActiveSetNew; ++k) {
                if (sot[k]->workSet_.any()) {
                    auto rows                = find(sot[k]->workSet_);
                    auto test                = (sot[k]->dual_(rows)).array() > tolerance;
                    sot[k]->activeSet_(rows) = !test;
                    sot[k]->lockedSet_(rows) = test;
                    isActiveSetNew           = k < k_ && test.any();
                }
            }
        }

        if (!isActiveSetNew) {
            h++;
        }
    }
}


void HierarchicalQP::dual_update(uint h, const Eigen::VectorXd& tau) {
    uint k      = 0;
    auto dof    = col_;
    auto oldDof = col_;

    while (k < h && dof > 0) {
        if (sot[k]->activeSet_.any()) {
            uint leftDof = dof - sot[k]->rank_;
            auto rows    = find(sot[k]->activeSet_ && sot[k]->workSet_);
            if (rows.any()) {
                Eigen::VectorXd f = inverse_.block(0, col_ - dof, oldDof, sot[k]->rank_).transpose() * tau.head(oldDof);
                sot[k]
                  ->codMid_.topLeftCorner(sot[k]->rank_, sot[k]->rank_)
                  .triangularView<Eigen::Upper>()
                  .transpose()
                  .solveInPlace<Eigen::OnTheLeft>(f);
                sot[k]->dual_(rows) = -sot[k]->codLeft_(rows, Eigen::seqN(0, sot[k]->rank_)) * f;
            }

            oldDof = dof;
            dof    = leftDof;
        }
        k++;
    }

    while (k < h) {
        auto rows = find(sot[k]->workSet_);
        sot[k]->dual_(rows).setZero();
        k++;
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


std::string HierarchicalQP::print_active_set() {
    std::stringstream out;
    out << "Active set:\n";
    for (uint k = 0; const auto& task : sot) {
        if (k < k_ && task->activeSet_.any()) {
            out << "\tLevel " << k << " -> constraints " << find(task->activeSet_).transpose() << "\n";
        }
        k++;
    }
    out << std::endl;
    return out.str();
}

}  // namespace hqp
