namespace hqp {

void TaskBase::set_mask(Eigen::VectorXi const& mask) {
    mask_ = mask;
}


template<typename... Args>
template<typename F>
Task<Args...>::Task(F&& f)
  : run_(std::forward<F>(f)) {
    // Matrix remains uninitialized (0x0) until parameters are provided
}


template<typename... Args>
void Task<Args...>::compute(Args... args) {
    std::tie(matrix, lower, upper) = run_(args...);

    assert(matrix.rows() == upper.rows());
    assert(lower.rows() == upper.rows());

    if (!mask_.size()) {
        mask_ = Eigen::VectorXi::Ones(matrix.cols());
    }
    if (weight_.size()) {
        // Weight subtasks within task
        matrix = weight_.matrixU() * matrix;
        lower  = weight_.matrixU() * lower;
        upper  = weight_.matrixU() * upper;
    }

    Eigen::MatrixXd tmp = matrix;
    matrix.resize(tmp.rows(), mask_.size());
    for (int k = 0, h = 0; h < mask_.size(); ++h) {
        if (mask_(h)) {
            matrix.col(h) = tmp.col(k++);
        }
    }
}


template<typename... Args, typename F>
auto bind_task(F&& f) {
    return std::make_shared<Task<Args...>>(std::forward<F>(f));
}


template<typename Derived>
Derived* TaskPtr::cast() {
    return static_cast<Derived*>(this->get());
}


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXi> StackOfTasks::get_stack() {
    if (this->empty()) {
        return {Eigen::MatrixXd(0, 0), Eigen::VectorXd(0), Eigen::VectorXd(0), Eigen::VectorXi(0)};
    }

    int rows = 0;
    int cols = -1;
    for (auto const& task : *this) {
        if (task->matrix.rows() <= 0) {
            throw std::runtime_error("Task not configured - call update() first");
        }
        assert((cols == -1 || task->matrix.cols() == cols) && "Inconsistent column size in tasks");
        cols  = task->matrix.cols();
        rows += task->matrix.rows();
    }

    Eigen::MatrixXd matrix(rows, cols);
    Eigen::VectorXd lower(rows), upper(rows);
    Eigen::VectorXi breaks(this->size());
    for (int start = 0, k = 0; auto const& task : *this) {
        rows = task->matrix.rows();

        matrix.middleRows(start, rows) = task->matrix;
        lower.segment(start, rows)     = task->lower;
        upper.segment(start, rows)     = task->upper;

        start       += rows;
        breaks(k++)  = start;
    }
    return {matrix, lower, upper, breaks};
}


void StackOfTasks::set_stack(Eigen::MatrixXd const& matrix,
                             Eigen::VectorXd const& lower,
                             Eigen::VectorXd const& upper,
                             Eigen::VectorXi const& breaks) {
    assert(matrix.rows() == upper.size() && upper.size() == lower.size() &&
           "matrix, upper, lower must have the same number of rows");
    assert(breaks.size() > 0 && "breaks must not be empty");
    for (int prev = 0, k = 0; k < breaks.size(); ++k) {
        assert(breaks[k] >= prev && "breaks must be increasing");
        prev = breaks[k];
    }
    assert(breaks(Eigen::last) == matrix.rows() && "The last break_point must be equal to matrix.rows()");

    if (breaks.size() != this->size()) {
        this->clear();
        this->resize(breaks.size());
    }
    for (int k = 0, start = 0; k < breaks.size(); ++k) {
        const int stop = breaks(k);
        const int rows = stop - start;

        // Optimal approach: Move the segments directly into the lambda
        // This avoids double copying while ensuring safety
        auto task = hqp::bind_task<>([matrix_seg = std::move(matrix.middleRows(start, rows).eval()),
                                      lower_seg  = std::move(lower.segment(start, rows).eval()),
                                      upper_seg  = std::move(upper.segment(start, rows).eval())]() mutable {
            return std::make_tuple(std::move(matrix_seg), std::move(lower_seg), std::move(upper_seg));
        });
        task->compute();
        this->at(k) = task;
        start = stop;
    }
}

}  // namespace hqp
