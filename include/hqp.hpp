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
#include "utils.hpp"

namespace hqp {

class HierarchicalQP {
  private:
    int row_;
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
    /** Current active lower-bound constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> activeLowSet_;
    /** Current active upper-bound constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> activeUpSet_;
    /** Initial equality constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
    /** Constraints temporarily locked. */
    Eigen::Array<bool, Eigen::Dynamic, 1> lockedSet_;
    /** Working set of constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> workSet_;
    Eigen::ArrayXi level_;
    /** Dual variables for inequality handling. */
    Eigen::VectorXd dual_;
    /** Right-hand side vector. */
    Eigen::VectorXd lower_;
    Eigen::VectorXd upper_;
    Eigen::VectorXd vector_;
    /** Constraint matrix computed by the task. */
    Eigen::MatrixXd matrix_;
    /** Auxiliary matrix for decomposition. */
    Eigen::MatrixXd codLefts_;

    /** Degrees of Freedom available for the task. */
    std::vector<int> dofs_;
    /** Rank of the task computed during solve. */
    std::vector<int> ranks_;
    /** Stores middle factor in decompositions. */
    std::vector<Eigen::MatrixXd> codMids_;
    /** Auxiliary matrix for decomposition. */
    std::vector<Eigen::MatrixXd> codRights_;
    /** Level of the nullspace in which to project. */
    // Eigen::ArrayXi parent_;

    /** Solves the overall HQP by combining tasks. */
    void solve();
    /** Handles equality constraint resolution. */
    void equality_hqp();
    /** Handles inequality constrained tasks. */
    void inequality_hqp();
    /** Updates dual variables. */
    void dual_update(int h);
    /** Increment primal due to contribution of active constraints in level. */
    void increment_primal(int parent, int level);

  public:
    /** Tolerance for convergence and numerical stability. */
    double tolerance = 1e-9;

    /**
     * @brief Constructs the HierarchicalQP solver.
     * @param n Number of degrees of freedom (columns) in the problem.
     */
    HierarchicalQP(int m, int n);

    /**
     * @brief Sets the metric matrix used to define the quadratic cost.
     * @param metric The metric matrix that influences the solver's behavior.
     */
    void set_metric(const Eigen::MatrixXd& metric);

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
    void set_problem(Eigen::MatrixXd const& matrix,
                     Eigen::VectorXd const& lower,
                     Eigen::VectorXd const& upper,
                     Eigen::VectorXi const& breaks);

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
