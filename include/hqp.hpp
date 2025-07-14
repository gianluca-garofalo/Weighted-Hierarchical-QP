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

namespace hqp {

// template<int MaxRows, int MaxCols>
class HierarchicalQP {
  private:
    int row_;
    /** Number of variables in the problem. */
    int col_;
    /** Number of levels in the task hierarchy. */
    int lev_;
    /** The current solution vector. */
    Eigen::VectorXd primal_;
    /** Intermediate task solution vector. */
    Eigen::VectorXd task_;
    /** Previous solution used as a warm start. */
    Eigen::VectorXd guess_;
    /** Force vector for primal updates. */
    Eigen::VectorXd force_;
    /** Stores the tau vector for dual updates. */
    Eigen::VectorXd tau_;
    /** Stores pseudo-inverse data. */
    Eigen::MatrixXd inverse_;
    /** Cholesky metric used for stability. */
    Eigen::MatrixXd cholMetric_;
    /** Reusable nullspace matrix to avoid re-instantiation. */
    Eigen::MatrixXd nullSpace_;

    /** Index tracking the active task level. */
    int k_ = 0;
    /** Current active lower-bound constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> activeLowSet_;
    /** Current active upper-bound constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> activeUpSet_;
    /** Initial equality constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
    /** Priority level for each constraint. */
    Eigen::ArrayXi level_;
    /** Dual variables for inequality handling. */
    Eigen::VectorXd dual_;
    /** Right-hand side vector. */
    Eigen::VectorXd lower_;
    Eigen::VectorXd upper_;
    Eigen::VectorXd vector_;
    /** Constraint matrix computed by the task. */
    Eigen::MatrixXd matrix_;
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::AutoAlign, MaxRows, MaxCols> matrix_;
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
    /** Index in sorted list of locked constraints for each task level. */
    std::vector<int> breaksEq1_;
    /** Index in sorted list of active constraints for each task level. */
    std::vector<int> breaksAc2_;
    /** Index in sorted list of inactive constraints for each task level. */
    Eigen::VectorXi breaks_;

    /** Solves the overall HQP by combining tasks. */
    void solve();
    /** Handles equality constraint resolution. */
    void equality_hqp();
    /** Handles inequality constrained tasks. */
    void inequality_hqp();
    /** Updates dual variables. */
    void dual_update(int h);
    /** Decrement solution after a change in the active set. */
    void decrement_from(int level);
    /** Increment solution after a change in the active set. */
    void increment_from(int level);
    /** Increment primal due to contribution of active constraints in level. */
    void increment_primal(int parent, int level);
    /** Locks a constraint at the specified row. */
    void lock_constraint(int row);
    /** Activates a constraint at the specified row, indicating whether it is a lower or upper bound. */
    void activate_constraint(int row, bool isLowerBound);
    /** Deactivates a constraint at the specified row. */
    void deactivate_constraint(int row);
    /** Swaps two constraints at specified indices. */
    void swap_constraints(int i, int j);

  public:
    /** Tolerance for convergence and numerical stability. */
    double tolerance = 1e-9;

    /**
     * @brief Constructs the HierarchicalQP solver.
     * @param n Number of degrees of freedom (columns) in the problem.
     */
    // template<int MaxRows, int MaxCols>
    // HierarchicalQP(int m, int n) {matrix_.resize(m, n);}
    HierarchicalQP(int m, int n);
    // HierarchicalQP(int m, int n) : HierarchicalQP<0, 0>(m, n) {}

    /**
     * @brief Sets the metric matrix used to define the quadratic cost.
     * @param metric The metric matrix that influences the solver's behavior.
     */
    void set_metric(const Eigen::MatrixXd& metric);

    /**
     * @brief Stacks multiple tasks into a TaskContainer for hierarchical QP.
     *
     * @param matrix          Constraint matrix (rows = total constraints, cols = variables)
     * @param lower           Lower bounds vector (size = total constraints)
     * @param upper           Upper bounds vector (size = total constraints)
     * @param breaks          Fixed-size vector of indices marking the end of each task in the stack
     *
     * The breaks vector specifies the cumulative constraint counts for exactly L levels.
     * For example, for L=3 levels with 2, 3, and 1 constraints respectively:
     * breaks = [2, 5, 6] (constraint indices: level 0: [0,2), level 1: [2,5), level 2: [5,6))
     *
     * Requirements:
     *   - matrix.rows() == lower.size() == upper.size()
     *   - breaks must be increasing and the last element equal to A.rows()
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

    // TODO: add get dual and get slack, where the latter are computed like in the test

    /**
     * @brief Outputs the active set details to the console.
     *
     * Provides a comprehensive summary of active constraints at each task level.
     */
    void print_active_set();
};

}  // namespace hqp

#endif  // _HierarchicalQP_
