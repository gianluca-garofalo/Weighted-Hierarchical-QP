#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include "library.hpp"

int main() {
    hqp::HierarchicalQP solver(2);
    auto task2 = hqp::SmartPtr<hqp::Task2>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto task3 = hqp::SmartPtr<hqp::Task3>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto task4 = hqp::SmartPtr<hqp::SubTasks>(Eigen::VectorXi::Ones(2).cast<bool>());
    auto task5 = hqp::SmartPtr<hqp::Task5>(Eigen::VectorXi::Ones(1).cast<bool>());
    auto task6 = hqp::SmartPtr<hqp::Task6>(Eigen::VectorXi::Ones(1).cast<bool>());
    task4->sot.push_back(task5);
    task4->sot.push_back(task6);
    solver.sot.push_back(task2);
    solver.sot.push_back(task3);
    solver.sot.push_back(task4);
    Eigen::Matrix<double, 2, 2> M;
    M << 10, 5, 5, 7;
    solver.set_metric(M);
    std::cout << "First Solution: " << solver.get_primal().transpose() << std::endl;

    std::swap(solver.sot[0], solver.sot[2]);
    task5->update();
    std::cout << "Updated Solution: " << solver.get_primal().transpose() << std::endl;

    return 0;
    return solver.get_primal().isApprox((Eigen::VectorXd(2) << 2.5, 1).finished()) &&
               solver.get_primal().isApprox(Eigen::VectorXd::Zero(2))
           ? 0
           : 1;
}
