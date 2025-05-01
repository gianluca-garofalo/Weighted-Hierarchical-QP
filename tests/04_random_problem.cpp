#include <chrono>
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <daqp.hpp>
#include <lexls/lexlsi.h>
#include <hqp.hpp>

LexLS::internal::LexLSI lexls_from_stack(Eigen::MatrixXd const& A,
                                         Eigen::VectorXd const& bu,
                                         Eigen::VectorXd const& bl,
                                         Eigen::VectorXi const& break_points) {
    auto n_tasks = break_points.size();
    std::vector<LexLS::Index> number_of_constraints(n_tasks);
    std::vector<LexLS::ObjectiveType> types_of_objectives(n_tasks);
    std::vector<Eigen::MatrixXd> objectives(n_tasks);

    for (unsigned int start = 0, k = 0; k < n_tasks; ++k) {
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
    lexls.setParameters(parameters);

    for (unsigned int i = 0; i < n_tasks; ++i) {
        lexls.setData(i, objectives[i]);
    }
    return lexls;
}


int main() {
    // Random problem generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> ncols_dist(2, 8);          // number of variables
    std::uniform_int_distribution<> ntasks_dist(1, 6);         // number of tasks
    std::uniform_int_distribution<> nrows_dist(1, 5);          // rows per task
    std::uniform_real_distribution<> val_dist(-10.0, 10.0);    // for matrix values
    std::uniform_real_distribution<> bound_dist(-20.0, 20.0);  // for bounds

    int ncols  = ncols_dist(gen);
    int ntasks = ntasks_dist(gen);
    std::vector<int> task_rows(ntasks);
    int total_rows = 0;
    for (int k = 0; k < ntasks; ++k) {
        task_rows[k]  = nrows_dist(gen);
        total_rows   += task_rows[k];
    }

    Eigen::MatrixXd A(total_rows + ncols, ncols);
    Eigen::VectorXd bu(total_rows + ncols);
    Eigen::VectorXd bl(total_rows + ncols);
    Eigen::VectorXi break_points(ntasks + 1);

    for (int start = 0, k = 0; k < ntasks; ++k) {
        int nrows = task_rows[k];
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                A(start + row, col) = val_dist(gen);
            }
            // Randomly decide if this constraint is equality or inequality
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < 0.3) {  // 30% chance equality
                double value    = bound_dist(gen);
                bl(start + row) = value;
                bu(start + row) = value;
            } else {  // 70% chance inequality
                double lower = bound_dist(gen);
                double upper = bound_dist(gen);
                if (lower > upper) {
                    std::swap(lower, upper);
                }
                bl(start + row) = lower;
                bu(start + row) = upper;
            }

            // Normalization due to how DAQP handles the problem
            auto norm           = A.row(start + row).norm();
            A.row(start + row) /= norm;
            bl(start + row)    /= norm;
            bu(start + row)    /= norm;
        }
        start           += nrows;
        break_points[k]  = start;
    }
    A.bottomRows(ncols)  = Eigen::MatrixXd::Identity(ncols, ncols);
    bu.tail(ncols)       = Eigen::VectorXd::Zero(ncols);
    bl.tail(ncols)       = Eigen::VectorXd::Zero(ncols);
    break_points(ntasks) = total_rows + ncols;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_stop  = t_start;

    // DAQP
    t_start     = std::chrono::high_resolution_clock::now();
    auto result = daqp_solve(A, bu, bl, (Eigen::VectorXi(1 + break_points.size()) << 0, break_points).finished());
    auto daqp   = result.get_primal();
    t_stop      = std::chrono::high_resolution_clock::now();
    std::cout << "Solution DAQP: " << daqp.transpose() << std::endl;
    std::cout << "DAQP execution time: " << std::chrono::duration<double>(t_stop - t_start).count() << " seconds"
              << std::endl;

    // HQP
    hqp::HierarchicalQP solver(ncols);
    solver.set_stack(A, bu, bl, break_points);
    t_start  = std::chrono::high_resolution_clock::now();
    auto hqp = solver.get_primal();
    t_stop   = std::chrono::high_resolution_clock::now();
    std::cout << "Solution HQP: " << hqp.transpose() << std::endl;
    std::cout << "HQP execution time: " << std::chrono::duration<double>(t_stop - t_start).count() << " seconds"
              << std::endl;

    // LEXLS
    auto tmp = lexls_from_stack(A, bu, bl, break_points);
    t_start  = std::chrono::high_resolution_clock::now();
    tmp.solve();
    auto lexls = tmp.get_x();
    t_stop     = std::chrono::high_resolution_clock::now();
    std::cout << "Solution LexLS: " << lexls.transpose() << std::endl;
    std::cout << "LexLS execution time: " << std::chrono::duration<double>(t_stop - t_start).count() << " seconds"
              << std::endl;

    double precision = 1e-5;
    return daqp.isApprox(hqp, precision) && daqp.isApprox(lexls, precision) ? 0 : 1;
}
