#include <lexls/lexlsi.h>

LexLS::internal::LexLSI lexls_from_stack(Eigen::MatrixXd const& A,
                                         Eigen::VectorXd const& bu,
                                         Eigen::VectorXd const& bl,
                                         Eigen::VectorXi const& break_points) {
    auto n_tasks = break_points.size();
    std::vector<LexLS::Index> number_of_constraints(n_tasks);
    std::vector<LexLS::ObjectiveType> types_of_objectives(n_tasks);
    std::vector<Eigen::MatrixXd> objectives(n_tasks);

    for (auto start = 0, k = 0; k < n_tasks; ++k) {
        auto n_constraints = break_points(k) - start;
        Eigen::MatrixXd objective(n_constraints, A.cols() + 2);
        objective << A.middleRows(start, n_constraints), bl.segment(start, n_constraints),
          bu.segment(start, n_constraints);
        number_of_constraints[k] = n_constraints;
        types_of_objectives[k]   = LexLS::ObjectiveType::GENERAL_OBJECTIVE;
        objectives[k]            = objective;
        start                    = break_points(k);
    }

    LexLS::internal::LexLSI lexls(A.cols(), n_tasks, &number_of_constraints[0], &types_of_objectives[0]);
    LexLS::ParametersLexLSI parameters;
    // parameters.output_file_name = "lexls_log.txt";
    lexls.setParameters(parameters);

    for (auto i = 0; i < n_tasks; ++i) {
        lexls.setData(i, objectives[i]);
    }
    return lexls;
}
