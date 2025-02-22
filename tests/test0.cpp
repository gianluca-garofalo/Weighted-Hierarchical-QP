#include <iostream>
#include <Eigen/Dense>
#include <hqp.hpp>
#include "library.hpp"


int main()
{
    hqp::HierarchicalQP solver(2);
    auto task0 = std::make_shared<hqp::Task0>(Eigen::VectorXi::Ones(1).cast<bool>());
    auto task1 = std::make_shared<hqp::Task1>(Eigen::VectorXi::Ones(2).cast<bool>());
    solver.sot.push_back(task0);
    solver.sot.push_back(task1);
    std::cout << "Solution: " << solver.get_primal().transpose() << std::endl;

    hqp::HierarchicalQP problem(2);
    std::shared_ptr<hqp::Task> task2 = std::make_shared<hqp::Task2>(Eigen::VectorXi::Zero(2).cast<bool>());
    std::shared_ptr<hqp::Task> task3 = std::make_shared<hqp::Task3>(Eigen::VectorXi::Zero(2).cast<bool>());
    std::shared_ptr<hqp::Task> task4 = std::make_shared<hqp::Task4>(Eigen::VectorXi::Ones(2).cast<bool>());
    for (const auto& task : { task2, task3, task4 })
    {
        problem.sot.push_back(task);
    }
    std::cout << "Solution: " << problem.get_primal().transpose() << std::endl;
    problem.print_active_set();

    std::swap(problem.sot[0], problem.sot[2]);
    dynamic_cast<hqp::Task4 *>(problem.sot[0].get())->update();
    std::cout << "Solution: " << problem.get_primal().transpose() << std::endl;
    problem.print_active_set();

    return 0;
}
