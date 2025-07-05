#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include <task.hpp>
#include "library.hpp"

int main() {
    hqp::StackOfTasks sot(3), task3_stack(2);

    task3_stack[0] = hqp::bind_task(run_task4);
    task3_stack[1] = hqp::bind_task(run_task5);
    task3_stack[0].cast<hqp::Task<>>()->compute();
    task3_stack[1].cast<hqp::Task<>>()->compute();

    sot[0] = hqp::bind_task(run_task2);
    sot[1] = hqp::bind_task(run_task3);
    sot[2] = hqp::bind_task<>([&task3_stack]() {
        auto [A, bl, bu, breaks] = task3_stack.get_stack();
        return std::make_tuple(A, bl, bu);
    });
    for (auto& task : sot) {
        task.cast<hqp::Task<>>()->compute();
    }

    Eigen::Matrix<double, 2, 2> M;
    M << 10, 5, 5, 7;

    auto [A, bl, bu, breaks] = sot.get_stack();
    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_metric(M);
    solver.set_problem(A, bl, bu, breaks);
    auto first_solution = solver.get_primal();
    std::cout << "First Solution: " << first_solution.transpose() << std::endl;

    std::swap(sot[0], sot[2]);
    std::tie(A, bl, bu, breaks) = sot.get_stack();
    solver.set_problem(A, bl, bu, breaks);
    auto second_solution = solver.get_primal();
    std::cout << "Second Solution: " << second_solution.transpose() << std::endl;

    return first_solution.isApprox(Eigen::Vector2d(2.5, 1)) && second_solution.isZero() ? 0 : 1;
}
