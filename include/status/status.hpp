#ifndef STATUS_STATUS_HPP
#define STATUS_STATUS_HPP

#include <iostream>

namespace hqp {

/**
 * @brief Status codes for the HQP solver.
 *
 * These codes indicate the outcome of the solve process.
 */

enum class SolverStatus {
    SUCCESS,                    ///< Solution found.
    MAX_ITERATIONS_REACHED,     ///< Maximum number of iterations reached.
    TIMEOUT_REACHED,            ///< Timeout reached.
    CYCLING_DETECTED,           ///< Solver detected cycling.
    STAGNATION_DETECTED,        ///< Solver detected stagnation.
    NUMERICAL_ISSUE,            ///< Numerical issue encountered.
    INFEASIBLE,                 ///< No solution exists for the problem.
    UNBOUNDED,                  ///< Solution is unbounded.
    NOT_SOLVED                  ///< Solver has not been run yet.
};

/**
 * @brief Detailed solver statistics and information.
 *
 * Contains information about the solve process, useful for debugging and performance analysis.
 */

struct SolverInfo {
    SolverStatus status = SolverStatus::NOT_SOLVED;  ///< Status code for the solve process.
    int total_iterations = 0;                        ///< Total number of iterations performed.
    int levels_completed = 0;                        ///< Number of hierarchy levels processed.
    int constraint_activations = 0;                  ///< Number of constraints activated.
    int constraint_deactivations = 0;                ///< Number of constraints deactivated.
    int cycling_events = 0;                          ///< Number of cycling events detected.
    int stagnation_events = 0;                       ///< Number of stagnation events detected.
    double solve_time_seconds = 0.0;                 ///< Time taken to solve (in seconds).
    double final_tolerance = 0.0;                    ///< Final tolerance achieved.
    double max_violation = 0.0;                      ///< Maximum constraint violation.


    // Clear all statistics
    void clear();

    // Print a summary of this SolverInfo to the given output stream
    void print(std::ostream& os = std::cout) const;

    // Returns a human-readable message for the current status
    std::string message() const;

    // Returns true if the solver status is SUCCESS
    bool is_solved() const;
};




} // namespace hqp

#include <status/status.tpp>
#endif // STATUS_STATUS_HPP
