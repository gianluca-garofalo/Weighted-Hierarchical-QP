#include <Eigen/Dense>
#include <chrono>
#include <daqp.hpp>
#include <hqp.hpp>
#include <iostream>
#include <lexls/lexlsi.h>


class Task0 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ =
          (Eigen::MatrixXd(6, 3) << Eigen::MatrixXd::Identity(3, 3), -Eigen::MatrixXd::Identity(3, 3)).finished();
        vector_ = (Eigen::VectorXd(6) << -Eigen::VectorXd::Ones(3), -Eigen::VectorXd::Ones(3)).finished();
    }

  public:
    Task0(Eigen::Array<bool, Eigen::Dynamic, 1> const& set)
      : TaskInterface(set) {
    }
};

class Task1 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = (Eigen::MatrixXd(1, 3) << -1, -1, -1).finished();
        vector_ = -Eigen::VectorXd::Ones(1);
    }

  public:
    Task1(Eigen::Array<bool, Eigen::Dynamic, 1> const& set)
      : TaskInterface(set) {
    }
};

class Task2 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = (Eigen::MatrixXd(1, 3) << 1, -1, 0).finished();
        vector_ = 0.5 * Eigen::VectorXd::Ones(1);
    }

  public:
    Task2(Eigen::Array<bool, Eigen::Dynamic, 1> const& set)
      : TaskInterface(set) {
    }
};

class Task3 : public hqp::TaskInterface<> {
  private:
    void run() override {
        matrix_ = (Eigen::MatrixXd(2, 3) << 3, 1, -1, -3, -1, 1).finished();
        vector_ = (Eigen::VectorXd(2) << 10, -20).finished();
    }

  public:
    Task3(Eigen::Array<bool, Eigen::Dynamic, 1> const& set)
      : TaskInterface(set) {
    }
};


int main() {
    // HQP
    hqp::HierarchicalQP hqp(3);
    hqp.sot.reserve(4);
    hqp.sot.emplace_back<Task0>(Eigen::VectorXi::Zero(6).cast<bool>());
    hqp.sot.emplace_back<Task1>(Eigen::VectorXi::Zero(1).cast<bool>());
    hqp.sot.emplace_back<Task2>(Eigen::VectorXi::Ones(1).cast<bool>());
    hqp.sot.emplace_back<Task3>(Eigen::VectorXi::Zero(2).cast<bool>());
    auto t_start  = std::chrono::high_resolution_clock::now();
    auto solution = hqp.get_primal();
    auto t_end    = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_elapsed = t_end - t_start;
    std::cout << "Solution HQP: " << solution.transpose() << std::endl;
    std::cout << "HQP execution time: " << t_elapsed.count() << " seconds" << std::endl;

    // DAQP
    // Task 0: -1 <= x <= 1
    Eigen::MatrixXd A0  = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd bu0 = Eigen::VectorXd::Ones(3);
    Eigen::VectorXd bl0 = -Eigen::VectorXd::Ones(3);

    // Task 1: x1+x2+x3 <= 1
    Eigen::MatrixXd A1  = (Eigen::MatrixXd(1, 3) << 1, 1, 1).finished();
    Eigen::VectorXd bu1 = Eigen::VectorXd::Ones(1);
    Eigen::VectorXd bl1 = Eigen::VectorXd::Constant(1, -DAQP_INF);

    // Task 2: x1 - x2 == 0.5
    Eigen::MatrixXd A2  = (Eigen::MatrixXd(1, 3) << 1, -1, 0).finished();
    Eigen::VectorXd bu2 = 0.5 * Eigen::VectorXd::Ones(1);
    Eigen::VectorXd bl2 = 0.5 * Eigen::VectorXd::Ones(1);

    // Task 3: 10 <= 3*x1+x2-x3 <= 20
    Eigen::MatrixXd A3  = (Eigen::MatrixXd(1, 3) << 3, 1, -1).finished();
    Eigen::VectorXd bu3 = 20 * Eigen::VectorXd::Ones(1);
    Eigen::VectorXd bl3 = 10 * Eigen::VectorXd::Ones(1);

    // Stack the tasks
    Eigen::MatrixXd A  = (Eigen::MatrixXd(6, 3) << A0, A1, A2, A3).finished();
    Eigen::VectorXd bu = (Eigen::VectorXd(6) << bu0, bu1, bu2, bu3).finished();
    Eigen::VectorXd bl = (Eigen::VectorXd(6) << bl0, bl1, bl2, bl3).finished();

    Eigen::VectorXi break_points = (Eigen::VectorXi(4) << 3, 4, 5, 6).finished();

    // Solve
    DAQP daqp(3, 50, 5);
    t_start = std::chrono::high_resolution_clock::now();
    daqp.solve(A, bu, bl, break_points);
    solution  = daqp.get_primal();
    t_end     = std::chrono::high_resolution_clock::now();
    t_elapsed = t_end - t_start;
    std::cout << "Solution DAQP: " << solution.transpose() << std::endl;
    std::cout << "DAQP execution time: " << t_elapsed.count() << " seconds" << std::endl;

    // LexLS
    std::vector<LexLS::Index> number_of_constraints;
    std::vector<LexLS::ObjectiveType> types_of_objectives;
    std::vector<Eigen::MatrixXd> objectives;

    unsigned int start = 0;
    for (unsigned int k = 0; k < break_points.size(); ++k) {
        auto n_constraints = break_points(k) - start;
        Eigen::MatrixXd objective(n_constraints, A.cols() + 2);
        objective << A.middleRows(start, n_constraints), bl.segment(start, n_constraints),
          bu.segment(start, n_constraints);
        number_of_constraints.push_back(n_constraints);
        types_of_objectives.push_back(LexLS::ObjectiveType::GENERAL_OBJECTIVE);
        objectives.push_back(objective);
        start = break_points(k);
    }

    LexLS::internal::LexLSI lexls(A.cols(), break_points.size(), &number_of_constraints[0], &types_of_objectives[0]);
    LexLS::ParametersLexLSI parameters;
    lexls.setParameters(parameters);

    for (unsigned int i = 0; i < break_points.size(); ++i) {
        lexls.setData(i, objectives[i]);
    }

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
