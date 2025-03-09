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


    void Task::select_variables(const Eigen::VectorXi& indices) { indices_ = indices; }


    SubTasks::SubTasks(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set) {}


    void SubTasks::compute()
    {
        sot[0]->compute();
        auto cols = sot[0]->matrix_.cols();
        auto rows = equalitySet_.size();
        matrix_.resize(rows, cols);
        vector_.resize(rows);
        indices_ = sot[0]->indices_;
        for (uint start = 0; const auto & task : sot)
        {
            task->compute();
            assert(cols == task->matrix_.cols());
            assert(indices_ == task->indices_);
            auto m = task->vector_.rows();
            matrix_.middleRows(start, m) = task->matrix_;
            vector_.segment(start, m) = task->vector_;
            start += m;
        }
        isComputed_ = true;
    }


    void SubTasks::set_weight(const Eigen::MatrixXd& weight)
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

} // namespace hqp
