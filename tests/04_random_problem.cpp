#include <chrono>
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <daqp.hpp>
#include <hqp.hpp>
#include "lexls_interface.hpp"

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

    std::stringstream sA, sbl, sbu, sbk;
    sA << "A << ";
    sbl << "bl << ";
    sbu << "bu << ";
    sbk << "break_points << ";
    int row, col;
    for (row = 0; row < A.rows() - 1; ++row) {
        for (col = 0; col < A.cols(); ++col) {
            sA << A(row, col) << ", ";
        }
        sbl << bl(row) << ", ";
        sbu << bu(row) << ", ";
    }
    for (col = 0; col < A.cols() - 1; ++col) {
        sA << A(row, col) << ", ";
    }
    sA << A(row, col) << ";" << std::endl;
    sbl << bl(row) << ";" << std::endl;
    sbu << bu(row) << ";" << std::endl;

    for (row = 0; row < break_points.rows() - 1; ++row) {
        sbk << break_points(row) << ", ";
    }
    sbk << break_points(row) << ";" << std::endl;

    std::cout << '\n' << sA.str() << sbl.str() << sbu.str() << sbk.str();

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

    return hqp.isApprox(lexls) ? 0 : 1;
}
