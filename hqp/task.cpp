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
        if (!is_computed_)
        {
            compute();
            if (has_guess_) vector_ -= matrix_ * guess_; // Shift problem to the origin
        }
        return vector_;
    }

    bool Task::set_equality(const Eigen::Array<bool, Eigen::Dynamic, 1>& is_equality)
    {
        auto m = is_equality.size();
        is_equality_ = is_equality;
        is_locked_ = is_free_ = Eigen::VectorXi::Zero(m).cast<bool>();
        slack_ = dual_ = Eigen::VectorXd::Zero(m);
        codMid_ = codLeft_ = Eigen::MatrixXd::Zero(m, m);
        return true;
    }

    bool Task::set_guess(const Eigen::VectorXd& guess)
    {
        has_guess_ = true;
        guess_ = guess;
        return true;
    }

} // namespace hqp
