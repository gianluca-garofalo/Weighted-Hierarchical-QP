#include "task.hpp"

namespace hqp {

Task::Task(int size) {
}

void Task::set_mask(const Eigen::VectorXi& indices) {
    indices_ = indices;
}

bool Task::is_computed() {
    return isComputed_;
}


void StackOfTasks::set_stack(Eigen::MatrixXd const& A,
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

    this->reserve(break_points.size());
    for (int start = 0; int const& stop : break_points) {
        this->emplace_back<GenericTask>(stop - start);
        this->back().cast<GenericTask>()->update(
          A.middleRows(start, stop - start), bl.segment(start, stop - start), bu.segment(start, stop - start));
        this->back()->compute();
        start = stop;
    }
}


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXi> StackOfTasks::get_stack() {
    int n_rows = 0;
    int n_cols = -1;
    Eigen::VectorXi breaks(this->size());
    for (int k = 0; auto const& task : *this) {
        task->compute();
        assert((n_cols < 0 ? true : task->matrix_.cols() == n_cols) &&
               "All tasks must have the same number of columns");
        n_cols       = task->matrix_.cols();
        n_rows      += task->lower_.size();
        breaks(k++)  = n_rows;
    }

    Eigen::VectorXd lower(n_rows);
    Eigen::VectorXd upper(n_rows);
    Eigen::MatrixXd matrix(n_rows, n_cols);

    for (int start = 0; auto const& task : *this) {
        auto dim                       = task->lower_.size();
        lower.segment(start, dim)      = task->lower_;
        upper.segment(start, dim)      = task->upper_;
        matrix.middleRows(start, dim)  = task->matrix_;
        start                         += dim;
    }

    return {matrix, lower, upper, breaks};
}

TaskPtr StackOfTasks::to_task() {
    auto [A, bl, bu, break_points] = this->get_stack();

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

    TaskPtr task = hqp::SmartPtr<GenericTask>(0);
    task.cast<GenericTask>()->update(A, bl, bu);
    task->compute();

    return std::move(task);
}

}  // namespace hqp
