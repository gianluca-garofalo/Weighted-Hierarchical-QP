/**
 * @file library.hpp
 * @brief Defines test task classes for the Weighted-Hierarchical-QP solver.
 *
 * This header provides several derived task classes that are used to test the solver.
 * Each task implements a custom run() function that returns a matrix and vector tailored for testing.
 */
#ifndef _LibraryOfTasks_
#define _LibraryOfTasks_

#include <Eigen/Dense>
#include "task.hpp"

namespace hqp {

class Task0 : public TaskInterface<> {
  private:
    void run() override;

  public:
    Task0(int size)
      : TaskInterface(size) {
    }
};

class Task1 : public TaskInterface<double, Eigen::VectorXd const&> {
  private:
    void run(double, Eigen::VectorXd const& vec) override;

  public:
    Task1(int size)
      : TaskInterface(size) {
    }
};

class Task2 : public TaskInterface<> {
  private:
    void run() override;

  public:
    Task2(int size)
      : TaskInterface(size) {
    }
};

class Task3 : public TaskInterface<> {
  private:
    void run() override;

  public:
    Task3(int size)
      : TaskInterface(size) {
    }
};

class Task5 : public TaskInterface<> {
  private:
    void run() override;

  public:
    Task5(int size)
      : TaskInterface(size) {
    }
};

class Task6 : public TaskInterface<> {
  private:
    void run() override;

  public:
    Task6(int size)
      : TaskInterface(size) {
    }
};

}  // namespace hqp

#endif  // _LibraryOfTasks_
