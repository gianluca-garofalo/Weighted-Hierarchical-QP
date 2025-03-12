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
  protected:
    std::tuple<std::decay_t<Args>...> args_; ///< Cached parameters for the run() function.  return run(unpacked...);
    /**
     * @brief Executes the task with provided parameters.
     * @param args The arguments necessary to compute the task's matrix and vector.
     * @return A tuple containing the computed matrix and vector.
     */
    virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run(Args... args) = 0;

    void compute() override {
        std::tie(matrix_, vector_) = std::apply([this](Args... unpacked) { return run(unpacked...); }, args_);
        assert(matrix_.rows() == vector_.rows());
        assert(equalitySet_.size() == vector_.rows());
        if (!indices_.size()) {
            auto n = matrix_.cols();
            indices_ = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
        }
        isComputed_ = true;
    }

  public:
    /// @brief Constructs the TaskInterface with the given equality set.
    /// @param set Boolean array specifying equality constraints.
    TaskInterface(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set) {
    }

    /**
     * @brief Updates task parameters and marks the task for re-computation.};
     * @param args New parameters for the task.
     */
    void update(Args... args) {
        args_ = std::make_tuple(args...);
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

} // namespace hqp

#endif // _Task_
