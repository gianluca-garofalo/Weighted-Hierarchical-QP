#include "task.hpp"

namespace hqp {

Task::Task(int size) {
    rows_ = size;
}

void Task::set_mask(const Eigen::VectorXi& indices) {
    indices_ = indices;
}

bool Task::is_computed() {
    return isComputed_;
}


TaskContainer set_stack(Eigen::MatrixXd const& A,
                        Eigen::VectorXd const& bu,
                        Eigen::VectorXd const& bl,
                        Eigen::VectorXi const& break_points) {
    assert(A.rows() == bu.size() && bu.size() == bl.size() && "A, bu, bl must have the same number of rows");
    assert(break_points.size() > 0 && "break_points must not be empty");
    int prev = 0;
    for (auto k = 0; k < break_points.size(); ++k) {
        assert(break_points[k] >= prev && "break_points must be increasing");
        prev = break_points[k];
    }
    assert(break_points(Eigen::last) == A.rows() && "The last break_point must be equal to A.rows()");

    class GenericTask : public hqp::TaskInterface<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> {
      private:
        void run(Eigen::MatrixXd matrix, Eigen::VectorXd lower, Eigen::VectorXd upper) override {
            matrix_ = std::move(matrix);
            lower_  = std::move(lower);
            upper_  = std::move(upper);
        }

      public:
        GenericTask(int size)
          : TaskInterface(size) {
        }
    };

    TaskContainer sot(break_points.size());
    for (int start = 0; int const& stop : break_points) {
        sot.emplace_back<GenericTask>(stop - start);
        sot.back().cast<GenericTask>()->update(
          A.middleRows(start, stop - start), bl.segment(start, stop - start), bu.segment(start, stop - start));
        sot.back()->compute();
        start = stop;
    }

    return sot;
}


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXi> get_stack(TaskContainer const& sot) {
    int n_rows = 0;
    int n_cols = -1;
    Eigen::VectorXi breaks(sot.size());
    for (int k = 0; auto const& task : sot) {
        task->compute();
        assert((n_cols < 0 ? true : task->matrix_.cols() == n_cols) &&
               "All tasks must have the same number of columns");
        n_cols       = task->matrix_.cols();
        n_rows      += task->lower_.size();
        breaks(k++)  = n_rows;
    }

    Eigen::VectorXd lower_(n_rows);
    Eigen::VectorXd upper_(n_rows);
    Eigen::MatrixXd matrix_(n_rows, n_cols);

    for (int start = 0; auto const &task : sot) {
        auto dim                        = task->lower_.size();
        lower_.segment(start, dim)      = task->lower_;
        upper_.segment(start, dim)      = task->upper_;
        matrix_.middleRows(start, dim)  = task->matrix_;
        start                          += dim;
    }

    return {matrix_, lower_, upper_, breaks};
}


SubTasks::SubTasks(int size)
  : Task(size) {
}

void SubTasks::compute() {
    sot[0]->compute();
    auto cols = sot[0]->matrix_.cols();
    matrix_.resize(rows_, cols);
    lower_.resize(rows_);
    upper_.resize(rows_);
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
