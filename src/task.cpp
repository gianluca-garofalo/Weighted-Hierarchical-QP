/**
 * @file task.cpp
 * @brief Implements the methods declared in task.hpp.
 *
 * This file provides the implementation of the abstract Task and SubTasks methods,
 * handling the initialization, variable selection, and subtask aggregation using
 * various linear algebra techniques.
 */

#include "task.hpp"

namespace hqp {

Task::Task(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) {
    /**
     * @brief Initializes a Task instance with a given set of equality constraints.
     *
     * Sets up the equality, locked, and work sets; and initializes slack and dual variables.
     *
     * @param set Boolean array specifying which constraints are treated as equality constraints.
     */
    auto m       = set.size();
    equalitySet_ = set;
    lockedSet_ = workSet_ = Eigen::VectorXi::Zero(m).cast<bool>();
    slack_ = dual_ = Eigen::VectorXd::Zero(m);
    codMid_ = codLeft_ = Eigen::MatrixXd::Zero(m, m);
}

void Task::select_variables(const Eigen::VectorXi& indices) {
    /**
     * @brief Assigns the indices of variables involved in this task.
     *
     * Helps keep track of which columns in the matrix correspond to active problem variables.
     *
     * @param indices Vector specifying the positions of selected variables.
     */
    indices_ = indices;
}

bool Task::is_computed() {
    return isComputed_;
}

SubTasks::SubTasks(const Eigen::Array<bool, Eigen::Dynamic, 1>& set)
  : Task(set) {
}

void SubTasks::compute() {
    /**
     * @brief Aggregates the results from all subtasks to form a composite solution.
     *
     * The method computes each subtask and concatenates their matrices and vectors,
     * ensuring consistency in dimensions and variable indices.
     */
    sot[0]->compute();
    auto cols = sot[0]->matrix_.cols();
    auto rows = equalitySet_.size();
    matrix_.resize(rows, cols);
    vector_.resize(rows);
    indices_ = sot[0]->indices_;
    for (unsigned int start = 0; const auto& task : sot) {
        task->compute();
        assert(cols == task->matrix_.cols());
        assert(indices_ == task->indices_);
        auto m                        = task->vector_.rows();
        matrix_.middleRows(start, m)  = task->matrix_;
        vector_.segment(start, m)     = task->vector_;
        start                        += m;
    }
    isComputed_ = true;
}

void SubTasks::set_weight(const Eigen::MatrixXd& weight) {
    /**
     * @brief Applies a weight matrix to adjust the subtasks' outputs.
     *
     * Uses a Cholesky decomposition to modify the task's matrix and vector,
     * enhancing numerical stability during the solution process.
     *
     * @param weight The metric matrix applied to the subtasks.
     */
    Eigen::LLT<Eigen::MatrixXd> lltOf(weight);
    assert(weight.isApprox(weight.transpose()) && lltOf.info() != Eigen::NumericalIssue);

    if (is_computed()) {
        weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(matrix_);
        weight_.matrixU().solveInPlace<Eigen::OnTheLeft>(vector_);
        matrix_ = lltOf.matrixU() * matrix_;
        vector_ = lltOf.matrixU() * vector_;
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
