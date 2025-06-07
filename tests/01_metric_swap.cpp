#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include "library.hpp"

int main() {
    hqp::HierarchicalQP solver(2);
    auto task2 = hqp::SmartPtr<hqp::Task2>(2);
    auto task3 = hqp::SmartPtr<hqp::Task3>(2);
    auto task4 = hqp::SmartPtr<hqp::SubTasks>(2);
    auto task5 = hqp::SmartPtr<hqp::Task5>(1);
    auto task6 = hqp::SmartPtr<hqp::Task6>(1);
    task4->sot.push_back(task5);
    task4->sot.push_back(task6);
    solver.sot.push_back(task2);
    solver.sot.push_back(task3);
    solver.sot.push_back(task4);
    Eigen::Matrix<double, 2, 2> M;
    M << 10, 5, 5, 7;
    solver.set_metric(M);
    auto first_solution = solver.get_primal();
    std::cout << "First Solution: " << first_solution.transpose() << std::endl;

    std::swap(solver.sot[0], solver.sot[2]);
    task5->update();
    auto second_solution = solver.get_primal();
    std::cout << "Second Solution: " << second_solution.transpose() << std::endl;

    return first_solution.isApprox(Eigen::Vector2d(2.5, 1)) && second_solution.isZero() ? 0 : 1;
}
