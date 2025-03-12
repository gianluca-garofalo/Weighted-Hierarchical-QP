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
    auto task0 = std::make_shared<hqp::Task0>(Eigen::VectorXi::Ones(1).cast<bool>());
    task0->select_variables(Eigen::VectorXi::Zero(1));
    auto task1 = std::make_shared<hqp::Task1>(Eigen::VectorXi::Ones(2).cast<bool>());
    task1->update(8, Eigen::VectorXd::Ones(2));
    solver.sot.push_back(task0);
    solver.sot.push_back(task1);
    std::cout << "Initial Solution: " << solver.get_primal().transpose() << std::endl;

    hqp::HierarchicalQP problem(2);
    auto task2 = std::make_shared<hqp::Task2>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto task3 = std::make_shared<hqp::Task3>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto task4 = std::make_shared<hqp::SubTasks>(Eigen::VectorXi::Ones(2).cast<bool>());
    auto task5 = std::make_unique<hqp::Task5>(Eigen::VectorXi::Ones(1).cast<bool>());
    auto task6 = std::make_unique<hqp::Task6>(Eigen::VectorXi::Ones(1).cast<bool>());
    task4->sot.push_back(std::move(task5));
    task4->sot.push_back(std::move(task6));
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
