#ifndef _HierarchicalQP_UTILITIES_TPP_
#define _HierarchicalQP_UTILITIES_TPP_

#include <iostream>

namespace hqp {

// TODO: upgrade to a logger keeping track of the active set
template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::print_active_set() {
    std::cout << "\n=== HQP Solver Active Set Information ===" << std::endl;
    std::cout << "Problem size: " << row_ << " rows, " << col_ << " columns, " << lev_ << " levels" << std::endl;
    std::cout << "Solver status: " << solver_info_.message() << std::endl;
    std::cout << "Solve time: " << solver_info_.solve_time_seconds << " seconds" << std::endl;
    std::cout << "Total iterations: " << solver_info_.total_iterations << std::endl;
    std::cout << "Levels completed: " << solver_info_.levels_completed << "/" << lev_ << std::endl;
    std::cout << "Constraint activations: " << solver_info_.constraint_activations << std::endl;
    std::cout << "Constraint deactivations: " << solver_info_.constraint_deactivations << std::endl;
    std::cout << "Max violation: " << solver_info_.max_violation << std::endl;

    if (solver_info_.cycling_events > 0) {
        std::cout << "Cycling events: " << solver_info_.cycling_events << std::endl;
    }
    if (solver_info_.stagnation_events > 0) {
        std::cout << "Stagnation events: " << solver_info_.stagnation_events << std::endl;
    }
    
    std::cout << "Active set:\n";
    for (int start = 0, k = 0; k < k_; ++k) {
        std::cout << "\tLevel " << k << ":\n";
        for (int row = start; row < breaksAct_(k); ++row) {
            std::cout << "\t\t" << lower_(row) << " < " << matrix_.row(row) << " < " << upper_(row) << "\n";
        }
        start = breaks_(k);
    }
    std::cout << "==========================================\n" << std::endl;
}

}  // namespace hqp

#endif  // _HierarchicalQP_UTILITIES_TPP_
