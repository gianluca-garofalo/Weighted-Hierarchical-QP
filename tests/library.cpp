#include "library.hpp"

namespace hqp {

void Task0::run() {
    matrix_ = Eigen::MatrixXd::Identity(1, 1);
    vector_ = Eigen::VectorXd::Zero(1);
}

void Task1::run(double b0, Eigen::VectorXd const& vec) {
    matrix_ = Eigen::MatrixXd::Identity(2, 2);
    vector_ = b0 * vec;
}

void Task2::run() {
    matrix_ = (Eigen::MatrixXd(2, 2) << 0.1, -1, 1, -1).finished();
    vector_ = (Eigen::VectorXd(2) << -0.55, 1.5).finished();
}

void Task3::run() {
    matrix_ = (Eigen::MatrixXd(2, 2) << -1, 0, -1, -1).finished();
    vector_ = (Eigen::VectorXd(2) << -2.5, 2).finished();
}

void Task5::run() {
    matrix_ = (Eigen::MatrixXd(1, 2) << 1, 0).finished();
    vector_ = Eigen::VectorXd::Zero(1);
}

void Task6::run() {
    matrix_ = (Eigen::MatrixXd(1, 2) << 0, 1).finished();
    vector_ = Eigen::VectorXd::Zero(1);
}

}  // namespace hqp
