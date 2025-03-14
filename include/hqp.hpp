/**
 * @file hqp.hpp
 * @brief Main interface for the Weighted-Hierarchical-QP solver.
 *
 * This file defines the HierarchicalQP class which encapsulates the algorithm to solve
 * prioritized quadratic programming problems. The solver aggregates multiple tasks and applies
 * iterative refinements, handling both equality and inequality constraints.
 */
#ifndef _HierarchicalQP_
#define _HierarchicalQP_

#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include "task.hpp"
// #include "options.hpp"

namespace hqp {

class HierarchicalQP {
  private:
    uint col_;                                             ///< Number of variables in the problem.
    Eigen::VectorXd primal_;                               ///< The current solution vector.
    Eigen::VectorXd task_;                                 ///< Intermediate task solution vector.
    Eigen::VectorXd guess_;                                ///< Previous solution used as a warm start.
    Eigen::MatrixXd inverse_;                              ///< Stores pseudo-inverse data.
    Eigen::MatrixXd nullSpace_;                            ///< Basis for the nullspace.
    Eigen::MatrixXd codRight_;                             ///< Matrix used for decomposition adjustments.
    Eigen::MatrixXd cholMetric_;                           ///< Cholesky metric used for stability.
    uint k_ = 0;                                           ///< Index tracking the active task level.

    void solve();                                          ///< Solves the overall HQP by combining tasks.
    void equality_hqp();                                   ///< Handles equality constraint resolution.
    void inequality_hqp();                                 ///< Handles inequality constrained tasks.
    void dual_update(uint h, const Eigen::VectorXd& tau);  ///< Updates dual variables.
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd> get_task(TaskPtr task,
                                                          const Eigen::VectorXi& row);  ///< Retrieves task data.

  public:
    double tolerance = 1e-9;                 ///< Tolerance for convergence and numerical stability.
    TaskContainer sot;                       ///< Container for all task pointers.

    /**
     * @brief Constructs the HierarchicalQP solver.
     * @param n Number of degrees of freedom (columns) in the problem.
     */
    HierarchicalQP(uint n);

    /**
     * @brief Sets the metric matrix used to define the quadratic cost.
     * @param metric The metric matrix that influences the solver's behavior.
     */
    void set_metric(const Eigen::MatrixXd& metric);

    /**
     * @brief Computes and retrieves the primal solution.
     * @return The computed primal solution vector.
     */
    Eigen::VectorXd get_primal();

    /**
     * @brief Outputs the active set details to the console.
     *
     * Provides a comprehensive summary of active constraints at each task level.
     */
    void print_active_set();
};

}  // namespace hqp

#endif  // _HierarchicalQP_
