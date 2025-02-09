#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>
#include <hqp/library.hpp>


int main()
{
    hqp::HierarchicalQP solver(2);
    auto task0 = std::make_shared<hqp::Task0>();
    auto task1 = std::make_shared<hqp::Task1>();
    task0->set_equality(Eigen::VectorXi::Ones(1).cast<bool>());
    task1->set_equality(Eigen::VectorXi::Ones(2).cast<bool>());
    solver.push_back(task0);
    solver.push_back(task1);
    std::cout << "Solution: " << solver.get_primal().transpose() << std::endl;

    hqp::HierarchicalQP problem(2);
    std::shared_ptr<hqp::Task> task2 = std::make_shared<hqp::Task2>();
    std::shared_ptr<hqp::Task> task3 = std::make_shared<hqp::Task3>();
    std::shared_ptr<hqp::Task> task4 = std::make_shared<hqp::Task4>();
    task2->set_equality(Eigen::VectorXi::Zero(2).cast<bool>());
    task3->set_equality(Eigen::VectorXi::Zero(2).cast<bool>());
    task4->set_equality(Eigen::VectorXi::Ones(2).cast<bool>());
    for (const auto& task : {task2, task3, task4})
    {
        problem.push_back(task);
    }
    std::cout << "Solution: " << problem.get_primal().transpose() << std::endl;

    return 0;
}
