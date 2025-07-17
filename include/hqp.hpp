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

template<int MaxRows = -1, int MaxCols = -1, int MaxLevels = -1, int ROWS = Eigen::Dynamic, int COLS = Eigen::Dynamic>
class HierarchicalQP {
  private:
    int row_;
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

    /** Index tracking the active task level. */
    int k_ = 0;
    /** Current active lower-bound constraints. */
    Eigen::Array<bool, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> activeLowSet_;
    /** Current active upper-bound constraints. */
    Eigen::Array<bool, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> activeUpSet_;
    /** Initial equality constraints. */
    Eigen::Array<bool, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> equalitySet_;
    /** Priority level for each constraint. */
    Eigen::Array<int, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> level_;
    /** Dual variables for inequality handling. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> dual_;
    /** Lower bounds for constraints. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> lower_;
    /** Upper bounds for constraints. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> upper_;
    /** Right-hand side vector. */
    Eigen::Matrix<double, ROWS, 1, Eigen::AutoAlign,MaxRows, 1> vector_;
    /** Constraint matrix computed by the task. */
    Eigen::Matrix<double, ROWS, COLS, Eigen::AutoAlign, MaxRows, MaxCols> matrix_;
    /** Left-hand side matrix in decompositions. */
    Eigen::Matrix<double, ROWS, ROWS, Eigen::AutoAlign, MaxRows, MaxRows> codLefts_;

    /** Degrees of Freedom available for the task. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign,MaxLevels, 1> dofs_;
    /** Ranks of the task computed during solve. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign,MaxLevels, 1> ranks_;
    /** Stores middle factor in decompositions. */
    std::vector<Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols>> codMids_;
    /** Stores right-hand side matrix in decompositions. */
    std::vector<Eigen::Matrix<double, COLS, COLS, Eigen::AutoAlign, MaxCols, MaxCols>> codRights_;
    /** Index in sorted list of locked constraints for each task level. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign,MaxLevels, 1> breaksFix_;
    /** Index in sorted list of active constraints for each task level. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign,MaxLevels, 1> breaksAct_;
    /** Index in sorted list of inactive constraints for each task level. */
    Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::AutoAlign,MaxLevels, 1> breaks_;

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

  public:
    /** Tolerance for convergence and numerical stability. */
    double tolerance = 1e-9;

    /**
     * @brief Constructs the HierarchicalQP solver.
     * @param n Number of degrees of freedom (columns) in the problem.
     */
    HierarchicalQP(int m, int n);

    /**
     * @brief Constructs the HierarchicalQP solver from an Eigen matrix with compile-time size.
     * @param matrix Eigen matrix with fixed compile-time dimensions
     * Template parameters are automatically deduced from the matrix dimensions.
     */
    template<int m, int n>
    HierarchicalQP(const Eigen::Matrix<double, m, n>& matrix);

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
     * 
     * This unified template function accepts both dynamic matrices (Eigen::MatrixXd, etc.)
     * and fixed-size matrices (Eigen::Matrix<double, m, n>, etc.) for optimal performance.
     * Compile-time checks ensure template parameter compatibility with fixed-size inputs.
     */
    template<typename MatrixType, typename LowerType, typename UpperType, typename BreaksType>
    void set_problem(const MatrixType& matrix,
                     const LowerType& lower,
                     const UpperType& upper,
                     const BreaksType& breaks);

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

};  // class HierarchicalQP

// Deduction guide for dynamic sizing (default behavior)
HierarchicalQP(int, int) -> HierarchicalQP<>;

// Deduction guide for Eigen matrix with compile-time size
template<int m, int n>
HierarchicalQP(const Eigen::Matrix<double, m, n>&) -> HierarchicalQP<m, n, -1, m, n>;

}  // namespace hqp


#include "hqp.tpp"

#endif  // _HierarchicalQP_
