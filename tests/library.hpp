#ifndef _LibraryOfTasks_
#define _LibraryOfTasks_

#include <functional>
#include <tuple>
#include <Eigen/Dense>
#include "task.hpp"

namespace hqp
{

    using Return = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
    template <typename... Args>
    using Map = std::pair<std::function<Return(Args...)>, std::tuple<Args...>>;

    class Task0 : public Task
    {
    private:
        Map<> map_;
        static Return run();

    public:
        Task0(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set), map_{ Map<>(run, std::make_tuple()) } {}
        void compute() override
        {
            std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
            assert(matrix_.rows() == vector_.rows());
            assert(equalitySet_.size() == vector_.rows());
            isComputed_ = true;
        }
    };


    class Task1 : public Task
    {
    private:
        Map<> map_;
        static Return run();

    public:
        Task1(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set), map_{ Map<>(run, std::make_tuple()) } {}
        void compute() override
        {
            std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
            assert(matrix_.rows() == vector_.rows());
            assert(equalitySet_.size() == vector_.rows());
            isComputed_ = true;
        }
        void update(int a, double b)
        {
            map_.second = std::make_tuple();
            isComputed_ = false;
        }
    };

    class Task2 : public Task
    {
    private:
        Map<> map_;
        static Return run();

    public:
        Task2(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set), map_{ Map<>(run, std::make_tuple()) } {}
        void compute() override
        {
            std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
            assert(matrix_.rows() == vector_.rows());
            assert(equalitySet_.size() == vector_.rows());
            isComputed_ = true;
        }
    };

    class Task3 : public Task
    {
    private:
        Map<> map_;
        static Return run();

    public:
        Task3(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set), map_{ Map<>(run, std::make_tuple()) } {}
        void compute() override
        {
            std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
            assert(matrix_.rows() == vector_.rows());
            assert(equalitySet_.size() == vector_.rows());
            isComputed_ = true;
        }
    };

    class Task4 : public Task
    {
    private:
        Map<> map_;
        static Return run();

    public:
        Task4(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set), map_{ Map<>(run, std::make_tuple()) } {}
        void compute() override
        {
            std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
            assert(matrix_.rows() == vector_.rows());
            assert(equalitySet_.size() == vector_.rows());
            isComputed_ = true;
        }
        void update()
        {
            map_.second = std::make_tuple();
            isComputed_ = false;
        }
    };

} // namespace hqp

#endif // _LibraryOfTasks_
