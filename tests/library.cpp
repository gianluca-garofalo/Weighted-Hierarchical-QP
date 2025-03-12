/**
 * @file library.cpp
 * @brief Implements the run() functions for various test tasks.
 *
 * Each task computes a specific matrix and vector to simulate different scenarios
 * for the hierarchical quadratic programming solver.
 */
#include "library.hpp"

namespace hqp {

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Task0::run() {
    return {Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Task1::run(double const& b0, Eigen::VectorXd const& vec) {
    return {Eigen::MatrixXd::Identity(2, 2), b0 * vec};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Task2::run() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
    A << -0.1, 1, -1, 1;
    b << 0.55, -1.5;
    return {A, b};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Task3::run() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
    A << 1, 0, 1, 1;
    b << 2.5, 2;
    return {A, b};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Task5::run() {
    return {(Eigen::MatrixXd(1, 2) << 1, 0).finished(), Eigen::VectorXd::Zero(1)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> Task6::run() {
    return {(Eigen::MatrixXd(1, 2) << 0, 1).finished(), Eigen::VectorXd::Zero(1)};
}

}  // namespace hqp
