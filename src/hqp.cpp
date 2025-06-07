#include <iostream>
#include "hqp.hpp"
#include "utils.hpp"

namespace hqp {

HierarchicalQP::HierarchicalQP(int n)
  : col_{n}
  , primal_{Eigen::VectorXd::Zero(n)}
  , task_{Eigen::VectorXd::Zero(n)}
  , guess_{Eigen::VectorXd::Zero(n)}
  , inverse_{Eigen::MatrixXd::Zero(n, n)}
  , nullSpace_{Eigen::MatrixXd::Identity(n, n)}
  , codRight_{Eigen::MatrixXd::Zero(n, n)}
  , cholMetric_{Eigen::MatrixXd::Identity(n, n)} {
}


void HierarchicalQP::solve() {
    bool isAllEquality = true;
    for (auto& task : sot) {
        prepare_task(task);
        if (!task->activeLowSet_.size() || !task->activeUpSet_.size()) {
            task->activeLowSet_ = task->activeUpSet_ = task->equalitySet_;
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
    for (auto k = k_; k < sot.size(); ++k) {
        auto rows = find(!sot[k]->equalitySet_);
        sot[k]->activeLowSet_(rows).setZero();
        sot[k]->activeUpSet_(rows).setZero();
        // TODO: it means equality constraints need to be added in ihqp.
    }
}


void HierarchicalQP::equality_hqp() {
    primal_.setZero();
    auto dof   = col_;
    nullSpace_ = cholMetric_;

    k_ = 0;
    for (; k_ < sot.size() && dof > 0; ++k_) {
        if ((sot[k_]->activeLowSet_ || sot[k_]->activeUpSet_).any()) {
            auto rows               = find(sot[k_]->activeLowSet_ || sot[k_]->activeUpSet_);
            Eigen::MatrixXd matrix  = sot[k_]->matrix_(rows, Eigen::all);
            Eigen::VectorXd vector  = sot[k_]->activeUpSet_(rows).select(sot[k_]->upper_(rows), sot[k_]->lower_(rows));
            vector                 -= matrix * primal_;

            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
            // TODO: dynamically update tolerances to avoid tasks oscillations
            cod.setThreshold(sot[k_]->tolerance);
            cod.compute(matrix * nullSpace_.leftCols(dof));
            auto rank    = cod.rank();
            auto leftDof = dof - rank;

            codRight_.leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation();
            if (leftDof > 0) {
                // In this case matrixZ() is not the identity, so Eigen computes it and is not garbage
                codRight_.leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
                nullSpace_.leftCols(leftDof) = codRight_.middleCols(rank, leftDof);
            }
            Eigen::MatrixXd codLeft = cod.householderQ();

            inverse_.middleCols(col_ - dof, rank) = codRight_.leftCols(rank);
            task_.segment(col_ - dof, rank)       = codLeft.leftCols(rank).transpose() * vector;
            sot[k_]->dual_(rows)                  = vector - codLeft.leftCols(rank) * task_.segment(col_ - dof, rank);
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
    }
}


void HierarchicalQP::set_metric(const Eigen::MatrixXd& metric) {
    Eigen::LLT<Eigen::MatrixXd> lltOf(metric);
    assert(metric.isApprox(metric.transpose()) && lltOf.info() != Eigen::NumericalIssue);
    cholMetric_.setIdentity();
    lltOf.matrixU().solveInPlace<Eigen::OnTheLeft>(cholMetric_);
}


void HierarchicalQP::set_stack(Eigen::MatrixXd const& A,
                               Eigen::VectorXd const& bu,
                               Eigen::VectorXd const& bl,
                               Eigen::VectorXi const& break_points) {
    assert(A.rows() == bu.size() && bu.size() == bl.size() && "A, bu, bl must have the same number of rows");
    assert(break_points.size() > 0 && "break_points must not be empty");
    int prev = 0;
    for (auto k = 0; k < break_points.size(); ++k) {
        assert(break_points[k] >= prev && "break_points must be increasing");
        prev = break_points[k];
    }
    assert(break_points(Eigen::last) == A.rows() && "The last break_point must be equal to A.rows()");

    class GenericTask : public hqp::TaskInterface<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> {
      private:
        void run(Eigen::MatrixXd matrix, Eigen::VectorXd lower, Eigen::VectorXd upper) override {
            matrix_ = std::move(matrix);
            lower_  = std::move(lower);
            upper_  = std::move(upper);
        }

      public:
        GenericTask(int size)
          : TaskInterface(size) {
        }
    };

    sot.reserve(break_points.size());
    for (int start = 0; int const& stop : break_points) {
        sot.emplace_back<GenericTask>(stop - start);
        sot.back().cast<GenericTask>()->update(
          A.middleRows(start, stop - start), bl.segment(start, stop - start), bu.segment(start, stop - start));
        sot.back()->compute();
        start = stop;
    }
}


Eigen::VectorXd HierarchicalQP::get_primal() {
    int k = 0;
    for (; k < k_ && sot[k]->is_computed(); ++k) {}
    if (k < k_ || k_ == 0) {
        solve();
    }
    return primal_;
}


void HierarchicalQP::inequality_hqp() {
    for (auto& task : sot) {
        task->lockedSet_.setZero();
    }
    Eigen::Index idx;
    int level, row;
    double slack, dual, mValue;
    bool isLowerBound;

    // TODO: replace maxIter with maxChanges for activations plus deactivations (each considered separately though)
    int maxIter = 500;
    for (auto iter = 0, h = 0; iter < maxIter && h < sot.size(); ++h) {
        slack = dual = 1;
        while ((slack > 0 || dual > 0) && iter < maxIter) {
            equality_hqp();

            // Add tasks to the active set.
            slack = -1;
            for (auto k = 0; k < sot.size(); ++k) {
                if ((!sot[k]->activeUpSet_).any()) {
                    auto rows              = find(!sot[k]->activeUpSet_);
                    Eigen::MatrixXd matrix = sot[k]->matrix_(rows, Eigen::all);
                    Eigen::VectorXd vector = sot[k]->upper_(rows);
                    mValue                 = (matrix * primal_ - vector).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        level        = k;
                        row          = rows(idx);
                        isLowerBound = false;
                    }
                }

                if ((!sot[k]->activeLowSet_).any()) {
                    auto rows              = find(!sot[k]->activeLowSet_);
                    Eigen::MatrixXd matrix = sot[k]->matrix_(rows, Eigen::all);
                    Eigen::VectorXd vector = sot[k]->lower_(rows);
                    mValue                 = (vector - matrix * primal_).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > slack) {
                        slack        = mValue;
                        level        = k;
                        row          = rows(idx);
                        isLowerBound = true;
                    }
                }
            }
            if (slack > tolerance) {
                if (isLowerBound) {
                    sot[level]->activeLowSet_(row) = true;
                } else {
                    sot[level]->activeUpSet_(row) = true;
                }
                continue;
            }

            // Remove tasks from the active set.
            dual_update(h);

            dual = -1;
            for (auto k = 0; k <= h; ++k) {
                sot[k]->workSet_ =
                  (sot[k]->activeLowSet_ || sot[k]->activeUpSet_) && !sot[k]->equalitySet_ && !sot[k]->lockedSet_;
                if (sot[k]->workSet_.any()) {
                    auto rows           = find(sot[k]->workSet_);
                    sot[k]->dual_(rows) = sot[k]->activeUpSet_(rows).select(sot[k]->dual_(rows), -sot[k]->dual_(rows));
                    mValue              = (sot[k]->dual_(rows)).maxCoeff(&idx);
                    if (mValue > tolerance && mValue > dual) {
                        dual  = mValue;
                        level = k;
                        row   = rows(idx);
                    }
                }
            }
            if (dual > tolerance) {
                sot[level]->activeLowSet_(row) = false;
                sot[level]->activeUpSet_(row)  = false;
                continue;
            }

            for (auto k = 0; k <= h; ++k) {
                if (sot[k]->workSet_.any()) {
                    auto rows                = find(sot[k]->workSet_);
                    sot[k]->lockedSet_(rows) = (sot[k]->dual_(rows)).array() < -tolerance;
                }
            }

            ++iter;
        }
    }
}


void HierarchicalQP::dual_update(int h) {
    auto rows              = find(sot[h]->activeLowSet_ || sot[h]->activeUpSet_);
    Eigen::MatrixXd matrix = sot[h]->matrix_(rows, Eigen::all);
    Eigen::VectorXd vector = sot[h]->activeUpSet_(rows).select(sot[h]->upper_(rows), sot[h]->lower_(rows));

    if (h >= k_) {
        sot[h]->dual_(rows) = vector - matrix * primal_;
        sot[h]->rank_       = 0;
    }
    Eigen::VectorXd tau = matrix.transpose() * sot[h]->dual_(rows);

    for (auto dof = sot[h]->rank_, k = h - 1; k >= 0; --k) {
        if ((sot[k]->activeLowSet_ || sot[k]->activeUpSet_).any()) {
            auto rows = find(sot[k]->activeLowSet_ || sot[k]->activeUpSet_);
            if (sot[k]->rank_ && k < k_) {
                dof               += sot[k]->rank_;
                Eigen::VectorXd f  = -inverse_.middleCols(col_ - dof, sot[k]->rank_).transpose() * tau;
                sot[k]
                  ->codMid_.topLeftCorner(sot[k]->rank_, sot[k]->rank_)
                  .triangularView<Eigen::Upper>()
                  .transpose()
                  .solveInPlace<Eigen::OnTheLeft>(f);
                sot[k]->dual_(rows)     = sot[k]->codLeft_(rows, Eigen::seqN(0, sot[k]->rank_)) * f;
                Eigen::MatrixXd matrix  = sot[k]->matrix_(rows, Eigen::all);
                tau                    += matrix.transpose() * sot[k]->dual_(rows);
            } else {
                sot[k]->dual_(rows).setZero();
            }
        }
    }
}


void HierarchicalQP::prepare_task(TaskPtr task) {
    if (!task->is_computed()) {
        task->compute();
        assert(task->indices_.maxCoeff() < col_);
        if (task->weight_.size()) {
            // Weight subtasks within task
            task->matrix_ = task->weight_.matrixU() * task->matrix_;
            task->lower_  = task->weight_.matrixU() * task->lower_;
            task->upper_  = task->weight_.matrixU() * task->upper_;
        }
        // Shift problem to the origin
        task->lower_ -= task->matrix_ * guess_(task->indices_);
        task->upper_ -= task->matrix_ * guess_(task->indices_);

        auto tmp = task->matrix_;
        task->matrix_.resize(tmp.rows(), col_);
        task->matrix_(Eigen::all, task->indices_) = tmp;
    }
}


// TODO: upgrade to a logger keeping track of the active set
void HierarchicalQP::print_active_set() {
    std::cout << "Active set:\n";
    for (auto k = 0; const auto& task : sot) {
        if (k < k_ && (task->activeLowSet_ || task->activeUpSet_).any()) {
            std::cout << "\tLevel " << k << " -> constraints "
                      << find(task->activeLowSet_ || task->activeUpSet_).transpose() << "\n";
        }
        k++;
    }
    std::cout << std::endl;
}

}  // namespace hqp
