#include "library.hpp"

namespace hqp
{

    Return Task0::run()
    {
        return { Eigen::MatrixXd::Identity(1, 2), Eigen::VectorXd::Zero(1) };
    }

    Return Task1::run()
    {
        return { Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Ones(2) };
    }

} // namespace hqp
