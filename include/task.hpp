/**
 * @file task.hpp
 * @brief Declares the abstract Task interface and its derivatives used in the HQP solver.
 *
 * The file defines the base Task class and the templated TaskInterface which allows passing
 * of different parameter types to the run() method. Also included is the SubTasks class which
 * aggregates multiple tasks into a composite unit.
 */
#ifndef _Task_
#define _Task_

#include <tuple>
#include <memory>
#include <utility>
#include <type_traits>
#include <cassert>
#include <optional>
#include <Eigen/Dense>
#include "utils.hpp"

namespace hqp {

class Task {
  private:
    int rows_;
    friend class SubTasks;

  protected:
    /** Indices of active variables. */
    Eigen::VectorXi indices_;
    /** Flag indicating if task computation is up-to-date. */
    bool isComputed_ = false;
    /** Weight based on Cholesky decomposition. */
    Eigen::LLT<Eigen::MatrixXd> weight_;

    virtual bool is_computed();

  public:
    /** Constraint matrix computed by the task. */
    Eigen::MatrixXd matrix_;
    /** Right-hand side vector. */
    Eigen::VectorXd lower_;
    Eigen::VectorXd upper_;

    /**
     * @brief Initializes a Task instance with a given set of equality constraints.
     *
     * Sets up the equality, locked, and work sets; and initializes slack and dual variables.
     *
     * @param set Boolean array specifying which constraints are treated as equality constraints.
     */
    Task(int size);
    virtual ~Task() = default;

    /**
     * @brief Assigns the indices of variables involved in this task.
     *
     * Helps keep track of which columns in the matrix correspond to active problem variables.
     *
     * @param indices Vector specifying the positions of selected variables.
     */
    void set_mask(const Eigen::VectorXi& indices);

    /**
     * @brief Pure virtual function to compute task-specific output.
     */
    virtual void compute() = 0;
};


/**
 * @brief Smart pointer alias for generic pointer types.
 * @tparam T The type to point to.
 */
template<typename T>
using SmartPtr      = GenericPtr<std::shared_ptr, T>;
/**
 * @brief Smart pointer alias for Task.
 */
using TaskPtr       = GenericPtr<std::shared_ptr, Task>;
/**
 * @brief Container for smart pointers to Task objects.
 */
using TaskContainer = SmartContainer<std::vector, GenericPtr, std::shared_ptr, Task>;


/**
 * @brief Stacks multiple tasks into a TaskContainer for hierarchical QP.
 *
 * @param A           Constraint matrix (rows = total constraints, cols = variables)
 * @param bu          Upper bounds vector (size = total constraints)
 * @param bl          Lower bounds vector (size = total constraints)
 * @param break_points Vector of indices marking the end of each task in the stack
 * @return TaskContainer with each task as a GenericTask
 *
 * Requirements:
 *   - A.rows() == bu.size() == bl.size()
 *   - break_points must be increasing and the last element equal to A.rows()
 */
TaskContainer set_stack(Eigen::MatrixXd const& A,
                        Eigen::VectorXd const& bu,
                        Eigen::VectorXd const& bl,
                        Eigen::VectorXi const& break_points);


std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXi> get_stack(TaskContainer const& sot);


/**
 * @brief Templated interface for tasks with arbitrary arguments.
 * @tparam Args Types of arguments required by the task.
 */
template<typename... Args>
class TaskInterface : public Task {
  private:
    template<typename T>
    struct is_reference_wrapper : std::false_type {};

    template<typename U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    // Unwrap a reference_wrapper; otherwise, forward the value
    template<typename T>
    static decltype(auto) unwrap(T&& t) {
        if constexpr (is_reference_wrapper<std::decay_t<T>>::value) {
            return t.get();
        } else {
            return std::forward<T>(t);
        }
    }

    // If T is a reference, store as reference_wrapper (to allow rebinding) otherwise, store by value
    template<typename T>
    using ArgStorage =
      std::conditional_t< std::is_reference<T>::value, std::reference_wrapper<std::remove_reference_t<T>>, T>;

  protected:
    // Use an optional tuple to delay initialization.
    std::optional<std::tuple<ArgStorage<Args>...>> args_;

    /**
     * @brief Executes the task using the provided parameters.
     * Derived classes must implement this function to compute and assign
     * the task's matrix_ and vector_ members.
     *
     * @param args The parameters necessary for the task.
     */
    virtual void run(Args... args) = 0;

    void compute() override {
        if (isComputed_) {
            return;
        }

        std::apply(
          [this](auto&&... args) {
              run(unwrap(std::forward<decltype(args)>(args))...);
          },
          *args_);

        assert(matrix_.rows() == upper_.rows());
        assert(lower_.rows() == upper_.rows());

        if (!indices_.size()) {
            indices_ = Eigen::VectorXi::Ones(matrix_.cols());
        }
        if (weight_.size()) {
            // Weight subtasks within task
            matrix_ = weight_.matrixU() * matrix_;
            lower_  = weight_.matrixU() * lower_;
            upper_  = weight_.matrixU() * upper_;
        }

        Eigen::MatrixXd tmp = matrix_;
        matrix_.resize(tmp.rows(), indices_.size());
        for (int k = 0, h = 0; h < indices_.size(); ++h) {
            if (indices_(h)) {
                matrix_.col(h) = tmp.col(k++);
            }
        }

        isComputed_ = true;
    }

  public:
    /**
     * @brief Constructs the TaskInterface with the specified equality set.
     * @param set Boolean array specifying equality constraints.
     *
     * Note: No task arguments are initialized here. You must call update() before compute().
     */
    TaskInterface(int size)
      : Task(size)
      , args_(std::nullopt) {
    }

    /**
     * @brief Updates task parameters and marks the task for re-computation.
     *
     * For reference parameters (or const references), the corresponding argument must be an lvalue.
     *
     * @param args New parameters for the task.
     */
    // If a parameter is a reference (or const reference), the corresponding argument must be an lvalue.
    template<
      typename... U,
      typename = std::enable_if_t< (sizeof...(U) == sizeof...(Args)) &&
                                   ((!std::is_reference<Args>::value || std::is_lvalue_reference<U>::value) && ...)>>
    void update(U&&... args) {
        args_       = std::tuple<ArgStorage<Args>...>(std::forward<U>(args)...);
        isComputed_ = false;
    }
};


// Composite task that aggregates multiple subtasks.
class SubTasks : public Task {
  protected:
    bool is_computed() override;

  public:
    /**
     * @brief Container holding smart pointers to subtasks.
     */
    TaskContainer sot;

    /**
     * @brief Constructs a SubTasks instance with a given equality constraint set.
     * @param set Boolean array indicating equality constraints.
     */
    SubTasks(int size);

    /**
     * @brief Aggregates the results from all subtasks to form a composite solution.
     *
     * The method computes each subtask and concatenates their matrices and vectors,
     * ensuring consistency in dimensions and variable indices.
     */
    void compute() override;

    /**
     * @brief Applies a weight matrix to adjust the subtasks' outputs.
     *
     * Uses a Cholesky decomposition to modify the task's matrix and vector,
     * enhancing numerical stability during the solution process.
     *
     * @param weight The metric matrix applied to the subtasks.
     */
    void set_weight(const Eigen::MatrixXd& weight);
};

}  // namespace hqp

#endif  // _Task_
