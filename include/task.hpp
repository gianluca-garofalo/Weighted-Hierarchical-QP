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

namespace hqp {

class Task {
  private:
    uint rank_;                                        ///< Rank of the task computed during solve.
    Eigen::VectorXd slack_;                            ///< Slack variables for constraint satisfaction.
    Eigen::VectorXd dual_;                             ///< Dual variables for inequality handling.
    Eigen::Array<bool, Eigen::Dynamic, 1> activeSet_;  ///< Current active constraints.
    Eigen::Array<bool, Eigen::Dynamic, 1> lockedSet_;  ///< Constraints temporarily locked.
    Eigen::Array<bool, Eigen::Dynamic, 1> workSet_;    ///< Working set of constraints.
    Eigen::MatrixXd codMid_;                           ///< Stores middle factor in decompositions.
    Eigen::MatrixXd codLeft_;                          ///< Auxiliary matrix for decomposition.
    friend class HierarchicalQP;
    friend class SubTasks;

  protected:
    Eigen::MatrixXd matrix_;                             ///< Constraint matrix computed by the task.
    Eigen::VectorXd vector_;                             ///< Right-hand side vector.
    Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;  ///< Initial equality constraints.
    Eigen::VectorXi indices_;                            ///< Indices of active variables.
    bool isComputed_ = false;                            ///< Flag indicating if task computation is up-to-date.
    Eigen::LLT<Eigen::MatrixXd> weight_;                 ///< Weight based on Cholesky decomposition.

    /// @brief Pure virtual function to compute task-specific output.
    virtual void compute() = 0;

  public:
    double tolerance = 1e-9;  ///< Tolerance value for computation accuracy.

    /**
     * @brief Constructs a Task by initializing constraint sets.
     * @param set Boolean array indicating equality constraints.
     */
    Task(const Eigen::Array<bool, Eigen::Dynamic, 1>& set);
    virtual ~Task() = default;

    /**
     * @brief Sets the indices of variables that are selected for this task.
     * @param indices Vector of selected variable indices.
     */
    void select_variables(const Eigen::VectorXi& indices);
};


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
          [this](auto &&...args) {run(unwrap(std::forward<decltype(args)>(args))...);},
          *args_);

        assert(matrix_.rows() == vector_.rows());
        assert(equalitySet_.size() == vector_.rows());

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
    TaskInterface(const Eigen::Array<bool, Eigen::Dynamic, 1>& set)
      : Task(set)
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
  public:
    /// @brief Container holding smart pointers to subtasks.
    std::vector<std::unique_ptr<Task>> sot;

    /// @brief Constructs a SubTasks instance with a given equality constraint set.
    /// @param set Boolean array indicating equality constraints.
    SubTasks(const Eigen::Array<bool, Eigen::Dynamic, 1>& set);

    /// @brief Computes the aggregated result from all subtasks.
    void compute() override;

    /// @brief Applies a weight matrix to adjust the combined outputs of the subtasks.
    /// @param weight The metric matrix used for scaling.
    void set_weight(const Eigen::MatrixXd& weight);

    template <typename T, typename... Args>
    void update(uint k, Args... args)
    {
        isComputed_ = false;
        static_cast<T*>(sot[k].get())->update(args...);
    }
};

}  // namespace hqp

#endif  // _Task_
