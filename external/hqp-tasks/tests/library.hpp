#ifndef _LibraryOfTasks_
#define _LibraryOfTasks_

#include <tuple>
#include <Eigen/Dense>

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task0();

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task1(double scale, Eigen::Vector2d const& offset);

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task2();

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task3();

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task4();

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> run_task5();

#endif  // _LibraryOfTasks_
