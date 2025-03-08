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
        weight_ = Eigen::LLT<Eigen::MatrixXd> (Eigen::MatrixXd::Identity(m, m));
    }


    void Task::set_weight(const Eigen::MatrixXd& weight)
    {
        Eigen::LLT<Eigen::MatrixXd> lltOf(weight);
        assert(weight.isApprox(weight.transpose()) && lltOf.info() != Eigen::NumericalIssue);

        if (isComputed_)
        {
            weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(matrix_);
            weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(vector_);
            matrix_ = lltOf.matrixU() * matrix_;
            vector_ = lltOf.matrixU() * vector_;
        }

        weight_ = lltOf;
    }

    void Task::select_variables(const Eigen::VectorXi& indices)
    {
        indices_ = indices;
    }

} // namespace hqp
