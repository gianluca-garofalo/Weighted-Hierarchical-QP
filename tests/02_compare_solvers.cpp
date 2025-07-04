#include <Eigen/Dense>
#include <chrono>
#include <daqp.hpp>
#include <hqp.hpp>
#include <task.hpp>
#include <iostream>
#include "lexls_interface.hpp"


class Task0 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = Eigen::MatrixXd::Identity(3, 3);
        lower_ = -Eigen::VectorXd::Ones(3);
        upper_ = Eigen::VectorXd::Ones(3);
    }

  public:
    Task0(int size)
      : TaskInterface(size) {
    }
};

class Task1 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = (Eigen::MatrixXd(1, 3) << 1, 1, 1).finished();
        lower_ = -1e9 * Eigen::VectorXd::Ones(1);
        upper_ = Eigen::VectorXd::Ones(1);
    }

  public:
    Task1(int size)
      : TaskInterface(size) {
    }
};

class Task2 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = (Eigen::MatrixXd(1, 3) << 1, -1, 0).finished();
        lower_ = upper_ = 0.5 * Eigen::VectorXd::Ones(1);
    }

  public:
    Task2(int size)
      : TaskInterface(size) {
    }
};

class Task3 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = (Eigen::MatrixXd(1, 3) << 3, 1, -1).finished();
        lower_ = 10 * Eigen::VectorXd::Ones(1);
        upper_ = 20 * Eigen::VectorXd::Ones(1);
    }

  public:
    Task3(int size)
      : TaskInterface(size) {
    }
};


int main() {
    // HQP
    hqp::StackOfTasks sot;
    sot.reserve(4);
    sot.emplace_back<Task0>(3);
    sot.emplace_back<Task1>(1);
    sot.emplace_back<Task2>(1);
    sot.emplace_back<Task3>(1);
    auto [A, bl, bu, break_points] = sot.get_stack();

    hqp::HierarchicalQP hqp(A.rows(), A.cols());
    hqp.set_problem(A, bl, bu, break_points);
    auto t_start  = std::chrono::high_resolution_clock::now();
    auto solution = hqp.get_primal();
    auto t_end    = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_elapsed = t_end - t_start;
    std::cout << "Solution HQP: " << solution.transpose() << std::endl;
    std::cout << "HQP execution time: " << t_elapsed.count() << " seconds" << std::endl;

    // DAQP
    DAQP daqp(3, 50, 5);
    t_start = std::chrono::high_resolution_clock::now();
    daqp.solve(A, bu, bl, break_points);
    solution  = daqp.get_primal();
    t_end     = std::chrono::high_resolution_clock::now();
    t_elapsed = t_end - t_start;
    std::cout << "Solution DAQP: " << solution.transpose() << std::endl;
    std::cout << "DAQP execution time: " << t_elapsed.count() << " seconds" << std::endl;

    // LexLS
    auto lexls  = lexls_from_stack(A, bu, bl, break_points);
    t_start     = std::chrono::high_resolution_clock::now();
    auto status = lexls.solve();
    solution    = lexls.get_x();
    t_end       = std::chrono::high_resolution_clock::now();
    t_elapsed   = t_end - t_start;
    std::cout << "Solution LexLS: " << solution.transpose() << std::endl;
    std::cout << "LexLS execution time: " << t_elapsed.count() << " seconds" << std::endl;

    double precision = 1e-5;
    return daqp.get_primal().isApprox(lexls.get_x(), precision) && hqp.get_primal().isApprox(lexls.get_x(), precision)
           ? 0
           : 1;
}
