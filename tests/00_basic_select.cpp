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
    std::cout << "Solution: " << solver.get_primal().transpose() << std::endl;

    return solver.get_primal().isApprox((Eigen::VectorXd(2) << 0, 8).finished())
           ? 0
           : 1;
}
