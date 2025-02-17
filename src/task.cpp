#include "task.hpp"

namespace hqp
{

    Task::Task(const Eigen::Array<bool, Eigen::Dynamic, 1>& set)
    {
        auto m = set.size();
        equalitySet_ = set;
        lockedSet_ = workSet_ = Eigen::VectorXi::Zero(m).cast<bool>();
        slack_ = dual_ = Eigen::VectorXd::Zero(m);
        codMid_ = codLeft_ = Eigen::MatrixXd::Zero(m, m);
    }

} // namespace hqp
