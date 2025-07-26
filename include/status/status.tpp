namespace hqp {

inline bool SolverInfo::is_solved() const {
    return status == SolverStatus::SUCCESS;
}

inline std::string SolverInfo::message() const {
    switch (status) {
        case SolverStatus::SUCCESS:
            return "Solution found successfully.";
        case SolverStatus::MAX_ITERATIONS_REACHED:
            return "Max iterations reached.";
        case SolverStatus::TIMEOUT_REACHED:
            return "Timeout reached.";
        case SolverStatus::CYCLING_DETECTED:
            return "Cycling detected.";
        case SolverStatus::STAGNATION_DETECTED:
            return "Stagnation detected.";
        case SolverStatus::NUMERICAL_ISSUE:
            return "Numerical issue encountered.";
        case SolverStatus::INFEASIBLE:
            return "Problem is infeasible.";
        case SolverStatus::UNBOUNDED:
            return "Problem is unbounded.";
        case SolverStatus::NOT_SOLVED:
            return "Not solved yet.";
        default:
            return "Unknown status.";
    }
}

inline void SolverInfo::clear() {
    status                   = SolverStatus::NOT_SOLVED;
    total_iterations         = 0;
    levels_completed         = 0;
    constraint_activations   = 0;
    constraint_deactivations = 0;
    cycling_events           = 0;
    stagnation_events        = 0;
    solve_time_seconds       = 0.0;
    final_tolerance          = 0.0;
    max_violation            = 0.0;
}

inline void SolverInfo::print(std::ostream& os) const {
    os << "Solver Status: " << message() << '\n';
    os << "Total Iterations: " << total_iterations << '\n';
    os << "Levels Completed: " << levels_completed << '\n';
    os << "Constraint Activations: " << constraint_activations << '\n';
    os << "Constraint Deactivations: " << constraint_deactivations << '\n';
    os << "Cycling Events: " << cycling_events << '\n';
    os << "Stagnation Events: " << stagnation_events << '\n';
    os << "Solve Time (s): " << solve_time_seconds << '\n';
    os << "Final Tolerance: " << final_tolerance << '\n';
    os << "Max Violation: " << max_violation << '\n';
}

} // namespace hqp
