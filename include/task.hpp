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
    /** Rank of the task computed during solve. */
    int rank_;
    /** Slack variables for constraint satisfaction. */
    Eigen::VectorXd slack_;
    /** Dual variables for inequality handling. */
    Eigen::VectorXd dual_;
    /** Current active constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> activeLowSet_;
    Eigen::Array<bool, Eigen::Dynamic, 1> activeUpSet_;
    /** Constraints temporarily locked. */
    Eigen::Array<bool, Eigen::Dynamic, 1> lockedSet_;
    /** Working set of constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> workSet_;
    /** Stores middle factor in decompositions. */
    Eigen::MatrixXd codMid_;
    /** Auxiliary matrix for decomposition. */
    Eigen::MatrixXd codLeft_;
    friend class HierarchicalQP;
    friend class SubTasks;

  protected:
    /** Constraint matrix computed by the task. */
    Eigen::MatrixXd matrix_;
    /** Right-hand side vector. */
    Eigen::VectorXd lower_;
    Eigen::VectorXd upper_;
    /** Initial equality constraints. */
    Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
    /** Indices of active variables. */
    Eigen::VectorXi indices_;
    /** Flag indicating if task computation is up-to-date. */
    bool isComputed_ = false;
    /** Weight based on Cholesky decomposition. */
    Eigen::LLT<Eigen::MatrixXd> weight_;

    /**
     * @brief Pure virtual function to compute task-specific output.
     */
    virtual void compute() = 0;
    virtual bool is_computed();

  public:
    /** Tolerance value for computation accuracy. */
    double tolerance = 1e-9;

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
    void select_variables(const Eigen::VectorXi& indices);
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
        std::apply(
          [this](auto&&... args) {
              run(unwrap(std::forward<decltype(args)>(args))...);
          },
          *args_);

        assert(matrix_.rows() == upper_.rows());
        assert(lower_.rows() == upper_.rows());
        assert(equalitySet_.size() == upper_.rows());


        equalitySet_ = upper_.array() == lower_.array();

        if (!indices_.size()) {
            auto n   = matrix_.cols();
            indices_ = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
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
