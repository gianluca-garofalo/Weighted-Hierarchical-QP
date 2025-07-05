#include <Eigen/Dense>
#include <chrono>
#include <daqp.hpp>
#include <hqp.hpp>
#include <task.hpp>
#include <iostream>
#include "lexls_interface.hpp"


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task0() {
    return {Eigen::MatrixXd::Identity(3, 3), -Eigen::VectorXd::Ones(3), Eigen::VectorXd::Ones(3)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task1() {
    return {(Eigen::MatrixXd(1, 3) << 1, 1, 1).finished(), -1e9 * Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task2() {
    return {
      (Eigen::MatrixXd(1, 3) << 1, -1, 0).finished(), 0.5 * Eigen::VectorXd::Ones(1), 0.5 * Eigen::VectorXd::Ones(1)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task3() {
    return {
      (Eigen::MatrixXd(1, 3) << 3, 1, -1).finished(), 10 * Eigen::VectorXd::Ones(1), 20 * Eigen::VectorXd::Ones(1)};
}


int main() {
    // Create a stack of tasks
    hqp::StackOfTasks sot(4);
    sot[0] = hqp::bind_task(run_task0);
    sot[1] = hqp::bind_task(run_task1);
    sot[2] = hqp::bind_task(run_task2);
    sot[3] = hqp::bind_task(run_task3);
    for (auto& task : sot) {
        task.cast<hqp::Task<>>()->compute();
    }
    auto [A, bl, bu, breaks] = sot.get_stack();

    // HQP
    hqp::HierarchicalQP hqp(A.rows(), A.cols());
    hqp.set_problem(A, bl, bu, breaks);
    auto t_start  = std::chrono::high_resolution_clock::now();
    auto solution = hqp.get_primal();
    auto t_end    = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_elapsed = t_end - t_start;
    std::cout << "Solution HQP: " << solution.transpose() << std::endl;
    std::cout << "HQP execution time: " << t_elapsed.count() << " seconds" << std::endl;

    // DAQP
    DAQP daqp(3, 50, 5);
    t_start = std::chrono::high_resolution_clock::now();
    daqp.solve(A, bu, bl, breaks);
    solution  = daqp.get_primal();
    t_end     = std::chrono::high_resolution_clock::now();
    t_elapsed = t_end - t_start;
    std::cout << "Solution DAQP: " << solution.transpose() << std::endl;
    std::cout << "DAQP execution time: " << t_elapsed.count() << " seconds" << std::endl;

    // LexLS
    auto lexls  = lexls_from_stack(A, bu, bl, breaks);
    t_start     = std::chrono::high_resolution_clock::now();
    auto status = lexls.solve();
    solution    = lexls.get_x();
    t_end       = std::chrono::high_resolution_clock::now();
    t_elapsed   = t_end - t_start;
    std::cout << "Solution LexLS: " << solution.transpose() << std::endl;
    std::cout << "LexLS execution time: " << t_elapsed.count() << " seconds" << std::endl;

    double precision = 1e-5;
    return daqp.get_primal().isApprox(lexls.get_x(), precision) && hqp.get_primal().isApprox(lexls.get_x(), precision)
           ? 0
           : 1;
}
