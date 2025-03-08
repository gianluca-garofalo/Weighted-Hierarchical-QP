#include "library.hpp"

namespace hqp
{

    Return Task0::run()
    {
        return { Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1) };
    }

    Return Task1::run()
    {
        return { Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Ones(2) };
    }

    Return Task2::run()
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
        A << -0.1, 1, -1, 1;
        b << 0.55, -1.5;
        return { A, b };
    }

    Return Task3::run()
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
        A << 1, 0, 1, 1;
        b << 2.5, 2;
        return { A, b };
    }

    Return Task4::run()
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
        A << 1, 0, 0, 1;
        b << 0, 0;
        return { A, b };
    }

} // namespace hqp
