#include "task.hpp"

namespace hqp {

Task::Task(int size) {
    equalitySet_ = lockedSet_ = workSet_ = Eigen::VectorXi::Zero(size).cast<bool>();
    enabledSet_ = activeSet_ = Eigen::VectorXi::Ones(size).cast<bool>();
    dual_ = Eigen::VectorXd::Zero(size);
    codMid_ = codLeft_ = Eigen::MatrixXd::Zero(size, size);
    parent_ = -Eigen::ArrayXi::Ones(size);
}

void Task::select_variables(const Eigen::VectorXi& indices) {
    indices_ = indices;
}

bool Task::is_computed() {
    return isComputed_;
}

SubTasks::SubTasks(int size)
  : Task(size) {
}

void SubTasks::compute() {
    sot[0]->compute();
    auto cols = sot[0]->matrix_.cols();
    auto rows = equalitySet_.size();
    matrix_.resize(rows, cols);
    lower_.resize(rows);
    upper_.resize(rows);
    indices_ = sot[0]->indices_;
    for (int start = 0; const auto& task : sot) {
        task->compute();
        assert(cols == task->matrix_.cols());
        assert(indices_ == task->indices_);
        auto m                        = task->lower_.rows();
        matrix_.middleRows(start, m)  = task->matrix_;
        lower_.segment(start, m)      = task->lower_;
        upper_.segment(start, m)      = task->upper_;
        start                        += m;
    }
    isComputed_ = true;
}

void SubTasks::set_weight(const Eigen::MatrixXd& weight) {
    Eigen::LLT<Eigen::MatrixXd> lltOf(weight);
    assert(weight.isApprox(weight.transpose()) && lltOf.info() != Eigen::NumericalIssue);

    if (is_computed()) {
        weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(matrix_);
        weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(lower_);
        weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(upper_);
        matrix_ = lltOf.matrixU() * matrix_;
        lower_  = lltOf.matrixU() * lower_;
        upper_  = lltOf.matrixU() * upper_;
    }

    weight_ = lltOf;
}

bool SubTasks::is_computed() {
    isComputed_ = true;
    for (const auto& task : sot) {
        isComputed_ &= task->isComputed_;
    }
    return isComputed_;
}

}  // namespace hqp
