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
#include "utils.hpp"

namespace hqp {

class HierarchicalQP {
  private:
    /** Number of variables in the problem. */
    int col_;
    /** The current solution vector. */
    Eigen::VectorXd primal_;
    /** Intermediate task solution vector. */
    Eigen::VectorXd task_;
    /** Previous solution used as a warm start. */
    Eigen::VectorXd guess_;
    /** Stores pseudo-inverse data. */
    Eigen::MatrixXd inverse_;
    /** Cholesky metric used for stability. */
    Eigen::MatrixXd cholMetric_;
    /** Index tracking the active task level. */
    int k_ = 0;

    void solve();                     ///< Solves the overall HQP by combining tasks.
    void equality_hqp();              ///< Handles equality constraint resolution.
    void inequality_hqp();            ///< Handles inequality constrained tasks.
    void dual_update(int h);          ///< Updates dual variables.
    void prepare_task(TaskPtr task);  ///< Prepares task data.

  public:
    /** Tolerance for convergence and numerical stability. */
    double tolerance = 1e-9;
    /** Container for all task pointers. */
    TaskContainer sot;

    /**
     * @brief Constructs the HierarchicalQP solver.
     * @param n Number of degrees of freedom (columns) in the problem.
     */
    HierarchicalQP(int n);

    /**
     * @brief Stacks multiple tasks into a TaskContainer for hierarchical QP.
     *
     * @param A           Constraint matrix (rows = total constraints, cols = variables)
     * @param bu          Upper bounds vector (size = total constraints)
     * @param bl          Lower bounds vector (size = total constraints)
     * @param break_points Vector of indices marking the end of each task in the stack
     * @return TaskContainer with each task as a GenericTask
     *
     * Requirements:
     *   - A.rows() == bu.size() == bl.size()
     *   - break_points must be increasing and the last element equal to A.rows()
     */
    void set_stack(Eigen::MatrixXd const& A,
                   Eigen::VectorXd const& bu,
                   Eigen::VectorXd const& bl,
                   Eigen::VectorXi const& break_points);

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
