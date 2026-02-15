/**
 * @file hqp.hpp
 * @brief Main interface for the HQP (Hierarchical Quadratic Programming) solver.
 *
 * Defines the `HierarchicalQP` class for solving prioritized quadratic programming problems.
 *
 * @author Gianluca Garofalo
 * @version 0.1.1
 * @date 2025
 */
#ifndef HQP_HIERARCHICALQP_HPP
#define HQP_HIERARCHICALQP_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>

namespace hqp {

template<int MaxRows   = -1,
         int MaxCols   = -1,
         int MaxLevels = -1,
         int ROWS      = Eigen::Dynamic,
         int COLS      = Eigen::Dynamic,
         int LEVS      = Eigen::Dynamic>
class HierarchicalQP {
  private:
    /** Number of variables in the problem. */
    int col_;
    /** Number of levels in the task hierarchy. */
    int lev_;
    /** The current solution vector. */
    Eigen::Matrix<double, COLS, 1, Eigen::AutoAlign, MaxCols, 1> primal_;
    /** Intermediate task solution vector. */
    Eigen::Matrix<double, COLS, 1, Eigen::AutoAlign, MaxCols, 1> task_;
    /** Previous solution used as a warm start. */
    Eigen::Matrix<double, COLS, 1, Eigen::AutoAlign, MaxCols, 1> guess_;
    /** Force vector for primal updates. */
    Eigen::Matrix<double, COLS, 1, Eigen::AutoAlign, MaxCols, 1> force_;
    /** Stores the tau vector for dual updates. */
    Eigen::Matrix<double, COLS, 1, Eigen::AutoAlign, MaxCols, 1> tau_;
    /** Stores pseudo-inverse data. */
    Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols> inverse_;
    /** Cholesky metric used for stability. */
    Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols> cholMetric_;
    /** Reusable nullspace matrix to avoid re-instantiation. */
    Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols> nullSpace_;
    /** Reusable COD decomposition to avoid repeated heap allocation. */
    Eigen::CompleteOrthogonalDecomposition<
      Eigen::Matrix<double, ROWS, COLS, Eigen::AutoAlign, MaxRows, MaxCols>> cod_;

    /** Index tracking the active task level. */
    int k_ = 0;
    /** Current active lower-bound constraints. */
    Eigen::Array<bool, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> activeLowSet_;
    /** Current active upper-bound constraints. */
    Eigen::Array<bool, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> activeUpSet_;
    /** Initial equality constraints. */
    Eigen::Array<bool, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> equalitySet_;
    /** Priority level for each constraint. */
    Eigen::Array<int, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> level_;
    /** Dual variables for inequality handling. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> dual_;
    /** Lower bounds for constraints. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> lower_;
    /** Upper bounds for constraints. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> upper_;
    /** Right-hand side vector. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> vector_;
    /** Cached slack variables for lower bounds (negative violations only). */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> slackLow_;
    /** Cached slack variables for upper bounds (positive violations only). */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> slackUp_;
    /** Constraint matrix computed by the task. */
    Eigen::Matrix<double, ROWS, COLS, Eigen::AutoAlign, MaxRows, MaxCols> matrix_;
    /** Left-hand side matrix in decompositions. */
    Eigen::Matrix<double, ROWS, ROWS, Eigen::AutoAlign, MaxRows, MaxRows> codLefts_;
    /** Row permutation tracking: perm_(i) = original row index of current row i. */
    Eigen::Matrix<int, ROWS, 1, Eigen::AutoAlign, MaxRows, 1> perm_;

    /** Flag indicating if slack variables are up-to-date. */
    bool slacksValid_ = false;
    /** Flag indicating if primal solution is up-to-date. */
    bool primalValid_ = false;

    /** Degrees of Freedom available for the task. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign, MaxLevels, 1> dofs_;
    /** Ranks of the task computed during solve. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign, MaxLevels, 1> ranks_;
    /** Stores middle factor in decompositions. */
    std::vector<Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols>> codMids_;
    /** Stores right-hand side matrix in decompositions. */
    std::vector<Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols>> codRights_;
    /** Index in sorted list of locked constraints for each task level. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign, MaxLevels, 1> breaksFix_;
    /** Index in sorted list of active constraints for each task level. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign, MaxLevels, 1> breaksAct_;
    /** Index in sorted list of inactive constraints for each task level. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign, MaxLevels, 1> breaks_;

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
    /** Computes the parent level for a given task level. */
    int get_parent(int level);
    /** Computes the squared constraint violation cost at a given level. */
    double get_level_cost(int k);


  public:
    /** Tolerance for convergence and numerical stability. */
    double tolerance = 1e-9;
    /** Number of active-set changes in the last solve (activations + deactivations). */
    int changes = 0;

    /**
     * @brief Constructs the HierarchicalQP solver.
     * @param m Number of constraints (rows) in your problem
     * @param n Number of variables (columns) in your problem
     *
     * Create the solver with the dimensions of your problem.
     */
    HierarchicalQP(int m, int n);

    /**
     * @brief Constructs the HierarchicalQP solver from an Eigen matrix with compile-time size.
     * @param matrix Eigen matrix with fixed compile-time dimensions
     * Template parameters are automatically deduced from the matrix dimensions.
     */
    template<int m, int n, int l>
    HierarchicalQP(const Eigen::Matrix<double, m, n>& matrix,
                   const Eigen::Vector<double, m>& lower,
                   const Eigen::Vector<double, m>& upper,
                   const Eigen::Vector<int, l>& breaks);

    /**
     * @brief Sets the metric matrix used to define the quadratic cost.
     * @param metric The metric matrix that influences the solver's behavior.
     */
    void set_metric(const Eigen::MatrixXd& metric);

    /**
     * @brief Stacks multiple tasks into a hierarchical QP problem.
     *
     * @param matrix          Constraint matrix (rows = total constraints, cols = variables)
     * @param lower           Lower bounds vector (size = total constraints)
     * @param upper           Upper bounds vector (size = total constraints)
     * @param breaks          Indices marking the end of each task level in the hierarchy
     *
     * The `breaks` vector defines your task hierarchy by marking the end of each priority level.
     *
     * Example: For 3 levels with 2, 3, and 1 constraints respectively:
     *   breaks = [2, 5, 6]  // Level 0: constraints [0,2), Level 1: [2,5), Level 2: [5,6)
     *
     * Requirements:
     * - `matrix.rows() == lower.size() == upper.size()`
     * - `breaks` must be increasing and end with `matrix.rows()`
     * - `lower <= upper` for all constraints
     *
     * Works with both dynamic and fixed-size Eigen matrices.
     */
    template<typename MatrixType, typename LowerType, typename UpperType, typename BreaksType>
    void set_problem(const MatrixType& matrix,
                     const LowerType& lower,
                     const UpperType& upper,
                     const BreaksType& breaks);

    /**
     * @brief Computes and retrieves the primal solution.
     * @return Const reference to the computed primal solution vector
     *
     * Call this method to solve your problem and get the solution. The solver will automatically run if it hasn't been
     * solved yet.
     *
     * What happens inside:
     * 1. If not already solved, triggers the solve process
     * 2. Applies hierarchical task resolution
     * 3. Returns the optimal solution vector
     * 4. Updates solver statistics
     */
    const Eigen::Matrix<double, COLS, 1, Eigen::AutoAlign, MaxCols, 1>& get_primal();

    /**
     * @brief Retrieves the slack variables.
     * @return The computed slack variables
     *
     * Call this method to get the slack variables after solving the problem. The solver will automatically run if it
     * hasn't been solved yet.
     */
    std::tuple<Eigen::Vector<double, ROWS>, Eigen::Vector<double, ROWS>> get_slack();

    /**
     * @brief Outputs the active set details to a stream.
     * @param os Output stream (defaults to std::cout)
     *
     * Provides a comprehensive summary of active constraints at each task level.
     */
    void print_active_set(std::ostream& os = std::cout);

};  // class HierarchicalQP

// Deduction guide for dynamic sizing (default behavior)
HierarchicalQP(int, int) -> HierarchicalQP<>;

}  // namespace hqp


#include "hqp.tpp"

#endif  // HQP_HIERARCHICALQP_HPP
