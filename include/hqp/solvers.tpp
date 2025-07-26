#ifndef _HierarchicalQP_SOLVERS_TPP_
#define _HierarchicalQP_SOLVERS_TPP_

#include <chrono>
#include "options.hpp"

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::solve() {
    // Reset solver info for new solve
    solver_info_.clear();
    auto solve_start_time = std::chrono::high_resolution_clock::now();

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

    // Compute final statistics
    auto solve_end_time             = std::chrono::high_resolution_clock::now();
    solver_info_.solve_time_seconds = std::chrono::duration<double>(solve_end_time - solve_start_time).count();
    solver_info_.final_tolerance    = tolerance;

    // Set success status if not already set by inequality_hqp
    if (solver_info_.status == SolverStatus::NOT_SOLVED) {
        solver_info_.status = SolverStatus::SUCCESS;
    }
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::equality_hqp() {
    primal_.setZero();
    k_ = std::numeric_limits<int>::max();
    increment_from(0);
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::inequality_hqp() {
    Eigen::Index idx;
    int row;
    double slack, dual, mValue;
    bool isLowerBound;  // needed to distinguish between upper and lower bound in case they are both active

    // Anti-cycling:
    frequency_.setOnes();

    equality_hqp();
    // TODO: replace maxIter with maxChanges for activations plus deactivations (each considered separately though)
    // Adaptive iteration limit based on problem size
    int maxIter               = std::min(HQP_MAX_ITERATIONS, 10 * row_ * lev_);  // Scale with problem size
    int consecutive_no_change = 0;
    int last_active_set_size  = 0;

    // Add timeout mechanism for very large problems
    auto start_time               = std::chrono::high_resolution_clock::now();
    const double max_time_seconds = HQP_TIMEOUT_SECONDS;  // Configurable timeout

    // Track cycling events
    int high_frequency_constraints = 0;

    for (auto iter = 0, h = 0; iter < maxIter && h < lev_; ++h) {
        slack = dual = 1;
        while ((slack > 0 || dual > 0) && iter < maxIter) {
            // Add tasks to the active set.
            slack = -1;
            for (int k = 0; k < lev_; ++k) {
                int dim = breaks_(k) - breaksAct_(k);
                if (dim > 0) {
                    vector_.segment(breaksAct_(k), dim).noalias() =
                      (!activeUpSet_.segment(breaksAct_(k), dim)).select(upper_.segment(breaksAct_(k), dim), 1e9);
                    mValue = ((matrix_.middleRows(breaksAct_(k), dim) * primal_ - vector_.segment(breaksAct_(k), dim))
                                .array() /
                              frequency_.segment(breaksAct_(k), dim))
                               .maxCoeff(&idx);

                    if (mValue * frequency_(breaksAct_(k) + idx) > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = breaksAct_(k) + idx;
                        isLowerBound = false;
                    }

                    vector_.segment(breaksAct_(k), dim).noalias() =
                      (!activeLowSet_.segment(breaksAct_(k), dim)).select(lower_.segment(breaksAct_(k), dim), -1e9);
                    mValue = ((vector_.segment(breaksAct_(k), dim) - matrix_.middleRows(breaksAct_(k), dim) * primal_)
                                .array() /
                              frequency_.segment(breaksAct_(k), dim))
                               .maxCoeff(&idx);

                    if (mValue * frequency_(breaksAct_(k) + idx) > tolerance && mValue > slack) {
                        slack        = mValue;
                        row          = breaksAct_(k) + idx;
                        isLowerBound = true;
                    }
                }
            }
            if (slack > tolerance) {
                frequency_(row) *= 5;
                decrement_from(level_(row));
                activate_constraint(row, isLowerBound);
                increment_from(level_(row));
                solver_info_.constraint_activations++;
                consecutive_no_change = 0;

                // Check for cycling
                if (frequency_(row) > 5) {
                    solver_info_.status         = SolverStatus::CYCLING_DETECTED;
                    // solver_info_.status_message = "Cycling detected: constraint " + std::to_string(row) +
                    //                              " activated " + std::to_string(frequency_(row)) + " times";
                    solver_info_.cycling_events = frequency_(row);
                }
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
                    dual_.segment(breaksFix_(k), dim).noalias() =
                      activeUpSet_.segment(breaksFix_(k), dim)
                        .select(dual_.segment(breaksFix_(k), dim), -dual_.segment(breaksFix_(k), dim));
                    mValue =
                      (dual_.segment(breaksFix_(k), dim).array() / frequency_.segment(breaksFix_(k), dim))
                        .maxCoeff(&idx);

                    if (mValue * frequency_(breaksFix_(k) + idx) > tolerance && mValue > dual) {
                        dual = mValue;
                        row  = breaksFix_(k) + idx;
                    }
                }
            }
            if (dual > tolerance) {
                frequency_(row) *= 5;
                decrement_from(level_(row));
                deactivate_constraint(row);
                increment_from(level_(row));
                solver_info_.constraint_deactivations++;
                consecutive_no_change = 0;
                continue;
            }

            for (int k = 0; k <= h; ++k) {
                for (int row = breaksFix_(k); row < breaksAct_(k); ++row) {
                    if (dual_(row) < -tolerance) {
                        lock_constraint(row);
                    }
                }
            }

            // Check for convergence stagnation
            int current_active_set_size = 0;
            for (int k = 0; k < lev_; ++k) {
                current_active_set_size += breaksAct_(k) - breaksFix_(k);
            }

            if (current_active_set_size == last_active_set_size) {
                consecutive_no_change++;
                if (consecutive_no_change > HQP_STAGNATION_THRESHOLD) {
                    // Force a constraint change to break stagnation
                    frequency_.setOnes();
                    consecutive_no_change = 0;
                    solver_info_.stagnation_events++;

                    // Check if we're stuck in stagnation
                    if (solver_info_.stagnation_events > 5) {
                        solver_info_.status = SolverStatus::STAGNATION_DETECTED;
                        // solver_info_.status_message = "Solver stagnated: no progress for " +
                        //                              std::to_string(consecutive_no_change) + " iterations";
                        return;
                    }
                }
            } else {
                consecutive_no_change = 0;
                last_active_set_size  = current_active_set_size;
            }

            // Check timeout
            auto current_time      = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();
            if (elapsed_seconds > max_time_seconds) {
                solver_info_.status = SolverStatus::TIMEOUT_REACHED;
                // solver_info_.status_message = "Timeout reached after " + std::to_string(elapsed_seconds) + "
                // seconds";
                return;
            }

            ++iter;
        }

        // Check if we reached max iterations
        if (iter >= maxIter) {
            solver_info_.status = SolverStatus::MAX_ITERATIONS_REACHED;
            // solver_info_.status_message = "Maximum iterations (" + std::to_string(maxIter) + ") reached without
            // convergence";
            return;
        }

        solver_info_.levels_completed = h + 1;
        solver_info_.total_iterations = iter;

        frequency_.setOnes();
    }
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
    tau_ = matrix_.middleRows(start, dim).transpose() * dual_.segment(start, dim);

    for (int dof = ranks_(h), k = h - 1; k >= 0; --k) {
        start = k == 0 ? 0 : breaks_(k - 1);
        dim   = breaksAct_(k) - start;
        if (dim > 0) {
            if (ranks_(k) && k < k_) {
                dof                              += ranks_(k);
                force_.head(ranks_(k)).noalias()  = -inverse_.middleCols(col_ - dof, ranks_(k)).transpose() * tau_;
                codMids_[k]
                  .topLeftCorner(ranks_(k), ranks_(k))
                  .template triangularView<Eigen::Upper>()
                  .transpose()
                  .template solveInPlace<Eigen::OnTheLeft>(force_.head(ranks_(k)));
                dual_.segment(start, dim).noalias() =
                  codLefts_.block(start, 0, dim, ranks_(k)) * force_.head(ranks_(k));
                tau_ += matrix_.middleRows(start, dim).transpose() * dual_.segment(start, dim);
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
            primal_  -= inverse_.middleCols(col_ - dofs_(k), ranks_(k)) * task_.segment(col_ - dofs_(k), ranks_(k));
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
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(tolerance);
    cod.compute(matrix_.middleRows(start, n_rows) * nullSpace_.leftCols(dof));
    ranks_(k)   = cod.rank();
    int leftDof = dof - ranks_(k);

    if (leftDof > 0) {
        // In this case matrixZ() is not the identity, so Eigen computes it and is not garbage
        codRights_[k].leftCols(dof).noalias() =
          nullSpace_.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
    } else {
        codRights_[k].leftCols(dof).noalias() = nullSpace_.leftCols(dof) * cod.colsPermutation();
    }
    codLefts_.block(start, 0, n_rows, n_rows).noalias() =
      cod.householderQ() * Eigen::MatrixXd::Identity(n_rows, n_rows);

    inverse_.middleCols(col_ - dof, ranks_(k)).noalias() = codRights_[k].leftCols(ranks_(k));
    task_.segment(col_ - dof, ranks_(k)).noalias() =
      codLefts_.block(start, 0, n_rows, ranks_(k)).transpose() * vector_.segment(start, n_rows);
    dual_.segment(start, n_rows).noalias() =
      vector_.segment(start, n_rows) -
      codLefts_.block(start, 0, n_rows, ranks_(k)) * task_.segment(col_ - dof, ranks_(k));
    cod.matrixT()
      .topLeftCorner(ranks_(k), ranks_(k))
      .template triangularView<Eigen::Upper>()
      .template solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, ranks_(k)));
    primal_ += inverse_.middleCols(col_ - dof, ranks_(k)) * task_.segment(col_ - dof, ranks_(k));

    codMids_[k].topLeftCorner(ranks_(k), ranks_(k)).noalias() = cod.matrixT().topLeftCorner(ranks_(k), ranks_(k));
}

}  // namespace hqp

#endif  // _HierarchicalQP_SOLVERS_TPP_
