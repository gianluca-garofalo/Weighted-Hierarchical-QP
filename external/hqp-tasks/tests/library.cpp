#include "library.hpp"

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task0() {
    return {Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task1(double scale, Eigen::Vector2d const& vec) {
    return {Eigen::MatrixXd::Identity(2, 2), scale * vec, scale * vec};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task2() {
    return {(Eigen::MatrixXd(2, 2) << 0.1, -1, 1, -1).finished(),
            -1e9 * Eigen::Vector2d::Ones(),
            (Eigen::VectorXd(2) << -0.55, 1.5).finished()};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task3() {
    return {(Eigen::MatrixXd(2, 2) << 1, 0, 1, 1).finished(),
            (Eigen::VectorXd(2) << 2.5, 2).finished(),
            1e9 * Eigen::Vector2d::Ones()};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task4() {
    return {Eigen::MatrixXd::Identity(1, 2), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1)};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task5() {
    return {(Eigen::MatrixXd(1, 2) << 0, 1).finished(), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1)};
}
