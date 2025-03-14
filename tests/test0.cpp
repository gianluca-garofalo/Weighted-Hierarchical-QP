/**
 * @file test0.cpp
 * @brief Test driver for validating the Hierarchical-QP solver functionality.
 *
 * This program sets up multiple tasks with varying configurations,
 * invokes the solver, and prints out the solution along with the active set details.
 * It demonstrates the usage of both standalone and composite tasks.
 */
#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include "library.hpp"

int main() {
    hqp::HierarchicalQP solver(2);
    solver.sot.reserve(2);
    solver.sot.emplace_back<hqp::Task0>(Eigen::VectorXi::Ones(1).cast<bool>());
    solver.sot.back()->select_variables(Eigen::VectorXi::Zero(1));
    solver.sot.emplace_back<hqp::Task1>(Eigen::VectorXi::Ones(2).cast<bool>());
    Eigen::VectorXd v = Eigen::VectorXd::Ones(2);
    solver.sot.back().cast<hqp::Task1>()->update(8, v);
    std::cout << "Initial Solution: " << solver.get_primal().transpose() << std::endl;

    hqp::HierarchicalQP problem(2);
    auto task2 = hqp::SmartPtr<hqp::Task2>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto task3 = hqp::SmartPtr<hqp::Task3>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto task4 = hqp::SmartPtr<hqp::SubTasks>(Eigen::VectorXi::Ones(2).cast<bool>());
    auto task5 = hqp::SmartPtr<hqp::Task5>(Eigen::VectorXi::Ones(1).cast<bool>());
    auto task6 = hqp::SmartPtr<hqp::Task6>(Eigen::VectorXi::Ones(1).cast<bool>());
    task4->sot.push_back(task5);
    task4->sot.push_back(task6);
    problem.sot.push_back(task2);
    problem.sot.push_back(task3);
    problem.sot.push_back(task4);
    Eigen::Matrix<double, 2, 2> M;
    M << 10, 5, 5, 7;
    problem.set_metric(M);
    std::cout << "Updated Solution: " << problem.get_primal().transpose() << std::endl;
    problem.print_active_set();

    std::swap(problem.sot[0], problem.sot[2]);
    task4->update<hqp::Task5>(0);
    // task6->update(); // Use unique_ptr, so that this is not possible
    std::cout << "Final Solution: " << problem.get_primal().transpose() << std::endl;
    problem.print_active_set();

    return 0;
}
