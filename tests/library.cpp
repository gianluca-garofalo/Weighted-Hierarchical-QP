#include "library.hpp"

namespace hqp {

void Task0::run() {
    matrix_ = Eigen::MatrixXd::Identity(1, 1);
    lower_ = upper_ = Eigen::VectorXd::Zero(1);
}

void Task1::run(double b0, Eigen::VectorXd const& vec) {
    matrix_ = Eigen::MatrixXd::Identity(2, 2);
    lower_ = upper_ = b0 * vec;
}

void Task2::run() {
    matrix_ = (Eigen::MatrixXd(2, 2) << 0.1, -1, 1, -1).finished();
    lower_ = -1e9 * Eigen::Vector2d::Ones();
    upper_ = (Eigen::VectorXd(2) << -0.55, 1.5).finished();
}

void Task3::run() {
    matrix_ = (Eigen::MatrixXd(2, 2) << 1, 0, 1, 1).finished();
    lower_ = (Eigen::VectorXd(2) << 2.5, 2).finished();
    upper_ = 1e9 * Eigen::Vector2d::Ones();
}

void Task5::run() {
    matrix_ = (Eigen::MatrixXd(1, 2) << 1, 0).finished();
    lower_ = upper_ = Eigen::VectorXd::Zero(1);
}

void Task6::run() {
    matrix_ = (Eigen::MatrixXd(1, 2) << 0, 1).finished();
    lower_ = upper_ = Eigen::VectorXd::Zero(1);
}

}  // namespace hqp
