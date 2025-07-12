#include <Eigen/Dense>
#include <chrono>
#include <daqp.hpp>
#include <hqp.hpp>
#include <task.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
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

double calculate_mean(const std::vector<double>& times) {
    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

double calculate_std(const std::vector<double>& times) {
    double mean = calculate_mean(times);
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    return std::sqrt(sq_sum / times.size() - mean * mean);
}

double calculate_coefficient_of_variation(const std::vector<double>& times) {
    double mean = calculate_mean(times);
    double std = calculate_std(times);
    return std / mean;
}

int main() {
    const int num_iterations = 1000;
    std::vector<double> hqp_times, daqp_times, lexls_times;
    
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

    // Initialize solvers
    hqp::HierarchicalQP hqp(6, 3);
    hqp.set_problem(A, bl, bu, breaks);
    
    DAQP daqp(3, 50, 5);
    
    auto lexls = lexls_from_stack(A, bu, bl, breaks);

    std::cout << "Running " << num_iterations << " iterations for timing analysis...\n\n";

    // Benchmark HQP
    for (int i = 0; i < num_iterations; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto solution = hqp.get_primal();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_elapsed = t_end - t_start;
        hqp_times.push_back(t_elapsed.count());
    }

    // Benchmark DAQP
    for (int i = 0; i < num_iterations; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        daqp.solve(A, bu, bl, breaks);
        auto solution = daqp.get_primal();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_elapsed = t_end - t_start;
        daqp_times.push_back(t_elapsed.count());
    }

    // Benchmark LexLS
    for (int i = 0; i < num_iterations; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto status = lexls.solve();
        auto solution = lexls.get_x();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_elapsed = t_end - t_start;
        lexls_times.push_back(t_elapsed.count());
    }

    // Statistical analysis
    std::cout << "Timing Analysis Results:\n";
    std::cout << "========================\n\n";
    
    std::cout << "HQP Solver:\n";
    std::cout << "  Mean: " << calculate_mean(hqp_times) << " seconds\n";
    std::cout << "  Std:  " << calculate_std(hqp_times) << " seconds\n";
    std::cout << "  CV:   " << calculate_coefficient_of_variation(hqp_times) << "\n";
    std::cout << "  Min:  " << *std::min_element(hqp_times.begin(), hqp_times.end()) << " seconds\n";
    std::cout << "  Max:  " << *std::max_element(hqp_times.begin(), hqp_times.end()) << " seconds\n\n";
    
    std::cout << "DAQP Solver:\n";
    std::cout << "  Mean: " << calculate_mean(daqp_times) << " seconds\n";
    std::cout << "  Std:  " << calculate_std(daqp_times) << " seconds\n";
    std::cout << "  CV:   " << calculate_coefficient_of_variation(daqp_times) << "\n";
    std::cout << "  Min:  " << *std::min_element(daqp_times.begin(), daqp_times.end()) << " seconds\n";
    std::cout << "  Max:  " << *std::max_element(daqp_times.begin(), daqp_times.end()) << " seconds\n\n";
    
    std::cout << "LexLS Solver:\n";
    std::cout << "  Mean: " << calculate_mean(lexls_times) << " seconds\n";
    std::cout << "  Std:  " << calculate_std(lexls_times) << " seconds\n";
    std::cout << "  CV:   " << calculate_coefficient_of_variation(lexls_times) << "\n";
    std::cout << "  Min:  " << *std::min_element(lexls_times.begin(), lexls_times.end()) << " seconds\n";
    std::cout << "  Max:  " << *std::max_element(lexls_times.begin(), lexls_times.end()) << " seconds\n\n";

    std::cout << "Coefficient of Variation (CV) indicates timing consistency:\n";
    std::cout << "  Lower CV = more consistent timing\n";
    std::cout << "  Higher CV = more variable timing\n";

    return 0;
}
