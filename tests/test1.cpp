// g++ main.cpp -I/usr/include/eigen3 -o main
// ./main

#include <iostream>
#include <hqp/hqp.hpp>

int main()
{
    Eigen::Array<int, 1, 2> starts_{0, 3};
    Eigen::Array<int, 1, 2> stops_{2, 6};
    Eigen::Array<int, 10, 1> idx_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Eigen::Array<bool, 10, 1> rows = Eigen::Array<int, 10, 1>::Zero().cast<bool>();

    for (auto i = 0; i < starts_.size(); ++i)
    {
        rows = rows || ((starts_(i) <= idx_) && (idx_ <= stops_(i)));
    }

    std::cout << "idx: " << rows.transpose() << std::endl;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(10, 10);
    std::cout << "A:\n"
              << A(hqp::find(rows), Eigen::all) << std::endl;

    hqp::HQP<2, 3> hqp(Eigen::Matrix<double, 2, 3>::Identity());
    std::cout << "A:\n"
              << hqp.get_A() << std::endl;

    hqp::HQP<3, 2> hqp2;
    std::cout << "A:\n"
              << hqp2.get_A() << std::endl;

    const int m = 2;
    hqp::HQP<m, m> hqp3;
    std::cout << "A:\n"
              << hqp3.get_A() << std::endl;

    return 0;
}
