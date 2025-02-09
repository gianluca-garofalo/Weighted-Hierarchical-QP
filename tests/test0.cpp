#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>
#include <hqp/library.hpp>


int main()
{
    hqp::HierarchicalQP solver(2);
    auto task0 = std::make_shared<hqp::Task0>();
    auto task1 = std::make_shared<hqp::Task1>();
    task0->is_equality_ = Eigen::VectorXi::Ones(1);
    task1->is_equality_ = Eigen::VectorXi::Ones(2);
    solver.push_back(task0);
    solver.push_back(task1);
    std::cout << "Solution: " << solver.get_primal().transpose() << std::endl;

    return 0;
}
