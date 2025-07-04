#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include "library.hpp"

int main() {
    hqp::StackOfTasks sot;
    sot.reserve(2);
    sot.emplace_back<hqp::Task0>(1);
    sot.back()->set_mask((Eigen::VectorXi(2) << 1, 0).finished());
    sot.emplace_back<hqp::Task1>(2);
    Eigen::VectorXd v = Eigen::VectorXd::Ones(2);
    sot.back().cast<hqp::Task1>()->update(8, v);
    auto [A, bl, bu, break_points] = sot.get_stack();

    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_problem(A, bl, bu, break_points);
    std::cout << "Solution: " << solver.get_primal().transpose() << std::endl;

    return solver.get_primal().isApprox((Eigen::VectorXd(2) << 0, 8).finished()) ? 0 : 1;
}
