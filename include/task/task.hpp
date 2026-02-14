#ifndef HQP_TASK_HPP
#define HQP_TASK_HPP

#include <functional>
#include <memory>
#include <vector>
#include <tuple>
#include <Eigen/Dense>

namespace hqp {

struct TaskBase {
  protected:
    Eigen::VectorXi mask_               = Eigen::VectorXi(0);
    Eigen::LLT<Eigen::MatrixXd> weight_ = Eigen::LLT<Eigen::MatrixXd>(Eigen::MatrixXd(0, 0));

  public:
    Eigen::MatrixXd matrix = Eigen::MatrixXd(0, 0);
    Eigen::VectorXd lower  = Eigen::VectorXd(0);
    Eigen::VectorXd upper  = Eigen::VectorXd(0);

    virtual ~TaskBase() = default;
    void set_mask(Eigen::VectorXi const& mask);
};


template<typename... Args>
struct Task : public TaskBase {
  private:
    std::function<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>(Args...)> run_;

  public:
    template<typename F>
    explicit Task(F&& f);

    void compute(Args... args);
};


// Always bind function only, user must call compute()
template<typename... Args, typename F>
auto bind_task(F&& f);


class TaskPtr : public std::shared_ptr<TaskBase> {
  public:
    using std::shared_ptr<TaskBase>::shared_ptr;

    template<typename Derived>
    Derived* cast();
};


struct StackOfTasks : public std::vector<TaskPtr> {
    using std::vector<TaskPtr>::vector;
        
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXi> get_stack();
    void set_stack(Eigen::MatrixXd const& matrix,
                   Eigen::VectorXd const& lower,
                   Eigen::VectorXd const& upper,
                   Eigen::VectorXi const& breaks);
};

}  // namespace hqp

#include "task.tpp"

#endif  // HQP_TASK_HPP
