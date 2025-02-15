#include "task.hpp"

namespace hqp
{

    Eigen::MatrixXd Task::get_matrix()
    {
        if (!isComputed_) compute();
        return matrix_;
    }

    Eigen::VectorXd Task::get_vector()
    {
        if (!isComputed_)
        {
            compute();
            if (hasGuess_) vector_ -= matrix_ * guess_; // Shift problem to the origin
        }
        return vector_;
    }

    bool Task::set_equality(const Eigen::Array<bool, Eigen::Dynamic, 1>& equalitySet)
    {
        auto m = equalitySet.size();
        equalitySet_ = equalitySet;
        lockedSet_ = workSet_ = Eigen::VectorXi::Zero(m).cast<bool>();
        slack_ = dual_ = Eigen::VectorXd::Zero(m);
        codMid_ = codLeft_ = Eigen::MatrixXd::Zero(m, m);
        return true;
    }

    bool Task::set_guess(const Eigen::VectorXd& guess)
    {
        hasGuess_ = true;
        guess_ = guess;
        return true;
    }

} // namespace hqp
