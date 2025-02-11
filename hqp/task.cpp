#include "task.hpp"

namespace hqp
{

    Eigen::MatrixXd Task::get_matrix()
    {
        if (!is_computed_) compute();
        return matrix_;
    }

    Eigen::VectorXd Task::get_vector()
    {
        if (!is_computed_) compute();
        return vector_;
    }

    bool Task::set_equality(const Eigen::Array<bool, Eigen::Dynamic, 1>& is_equality)
    {
        is_equality_ = is_equality;
        auto m = is_equality.size();
        slack_ = Eigen::VectorXd::Zero(m);
        dual_ = Eigen::VectorXd::Zero(m);
        is_locked_ = Eigen::VectorXi::Zero(m).cast<bool>();
        is_free_ = Eigen::VectorXi::Zero(m).cast<bool>();
        codMid_ = Eigen::MatrixXd::Zero(m, m);
        codLeft_ = Eigen::MatrixXd::Zero(m, m);
        return true;
    }

    bool Task::set_guess(const Eigen::VectorXd& guess)
    {
        has_guess_ = true;
        guess_ = guess;
        return true;
    }

} // namespace hqp
