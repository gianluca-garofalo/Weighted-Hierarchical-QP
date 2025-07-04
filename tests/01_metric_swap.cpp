#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include "library.hpp"

int main() {
    auto task2 = hqp::SmartPtr<hqp::Task2>(2);
    auto task3 = hqp::SmartPtr<hqp::Task3>(2);
    
    hqp::StackOfTasks task4(2);
    auto task5 = hqp::SmartPtr<hqp::Task5>(1);
    task4[1] = hqp::SmartPtr<hqp::Task6>(1);
    task4[0] = task5;

    hqp::StackOfTasks sot;
    sot.push_back(task2);
    sot.push_back(task3);
    sot.push_back(task4.to_task());
    auto [A, bl, bu, break_points] = sot.get_stack();

    Eigen::Matrix<double, 2, 2> M;
    M << 10, 5, 5, 7;
    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_metric(M);
    solver.set_problem(A, bl, bu, break_points);
    auto first_solution = solver.get_primal();
    std::cout << "First Solution: " << first_solution.transpose() << std::endl;

    std::swap(sot[0], sot[2]);
    task5->update();
    std::tie(A, bl, bu, break_points) = sot.get_stack();
    solver.set_problem(A, bl, bu, break_points);
    auto second_solution = solver.get_primal();
    std::cout << "Second Solution: " << second_solution.transpose() << std::endl;

    return first_solution.isApprox(Eigen::Vector2d(2.5, 1)) && second_solution.isZero() ? 0 : 1;
}
