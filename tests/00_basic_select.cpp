#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>
#include <task/task.hpp>
#include "library.hpp"

int main() {
    hqp::StackOfTasks sot;
    sot.reserve(2);

    // Always two steps: 1) create task with function, 2) call update()
    auto task0 = hqp::bind_task<>(run_task0);
    task0->set_mask((Eigen::VectorXi(2) << 1, 0).finished());
    task0->compute();

    auto task1        = hqp::bind_task<double, Eigen::Vector2d>(run_task1);
    Eigen::VectorXd v = Eigen::VectorXd::Ones(2);
    task1->compute(1, 0 * v);
    task1->compute(8, v);

    sot.push_back(task0);
    sot.push_back(task1);

    auto [A, bl, bu, breaks] = sot.get_stack();
    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_problem(A, bl, bu, breaks);
    auto solution = solver.get_primal();
    std::cout << "Solution: " << solution.transpose() << std::endl;

    return solver.get_primal().isApprox((Eigen::VectorXd(2) << 0, 8).finished()) ? 0 : 1;
}
