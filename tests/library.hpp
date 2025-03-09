#ifndef _LibraryOfTasks_
#define _LibraryOfTasks_

#include <Eigen/Dense>
#include "task.hpp"

namespace hqp
{

    class Task0 : public TaskInterface<>
    {
    private:
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run() override;
    public:
        Task0(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : TaskInterface(set) {}
    };


    class Task1 : public TaskInterface<double, double>
    {
    private:
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run(double b0, double b1) override;
    public:
        Task1(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : TaskInterface(set) {}
    };


    class Task2 : public TaskInterface<>
    {
    private:
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run() override;
    public:
        Task2(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : TaskInterface(set) {}
    };


    class Task3 : public TaskInterface<>
    {
    private:
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run() override;
    public:
        Task3(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : TaskInterface(set) {}
    };

    
    class Task4 : public TaskInterface<>
    {
    private:
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run() override;
    public:
        Task4(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : TaskInterface(set) {}
    };

} // namespace hqp

#endif // _LibraryOfTasks_
