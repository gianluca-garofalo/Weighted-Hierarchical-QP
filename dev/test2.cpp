// g++ test2.cpp -I/usr/include/eigen3 -o test2
// ./test2

/*
Architecture of a Stack of Tasks, where the actual task computation is delayed until the solver actually needs the data.
If that is not necessary, then only the class Task is needed without the virtual compute method. Those computations will
have to be done outside the solver for all the tasks, so that the members are filled before passing them to the solver.
*/

#include <iostream>
#include <functional>
#include <memory>
#include <Eigen/Dense>

// Generic interface class to be used within the solver to compute and store data if the task gets activated.
class Task
{
protected:
    Eigen::MatrixXd matrix_;
    Eigen::VectorXd vector_;

public:
    Task() {}
    virtual void compute() = 0;
    void print()
    {
        std::cout << "Matrix:\n"
                  << matrix_ << std::endl;
        std::cout << "Vector:\n"
                  << vector_.transpose() << std::endl;
    }
};

// Specific task classes to be used outside the solver to update the inputs.
using Return = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
template <typename... Args>
using Map = std::pair<std::function<Return(Args...)>, std::tuple<Args...>>;

class Task0 : public Task
{
private:
    Map<int> map_;
    static Return run(int a)
    {
        return {Eigen::MatrixXd::Identity(a, a), Eigen::VectorXd::Zero(a)};
    }

public:
    Task0(int a) : Task(), map_{Map<int>(run, std::make_tuple(a))} {}
    void compute() override
    {
        std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
    }
};

class Task1 : public Task
{
private:
    Map<int, double> map_;
    static Return run(int a, double b)
    {
        return {Eigen::MatrixXd::Identity(a, a), b * Eigen::VectorXd::Ones(a)};
    }

public:
    Task1(int a, double b) : Task(), map_{Map<int, double>(run, std::make_tuple(a, b))} {}
    void compute() override
    {
        std::tie(matrix_, vector_) = std::apply(map_.first, map_.second);
    }
    void update(int a, double b)
    {
        map_.second = std::make_tuple(a, b);
    }
};

int main()
{
    // Create the Stack of Tasks.
    std::vector<std::unique_ptr<Task>> sot;
    sot.push_back(std::make_unique<Task1>(2, 3.14));
    sot.push_back(std::make_unique<Task0>(2));

    // Compute is used within the solver, where it is only known that tasks are expressed via matrix and vector.
    for (const auto &task : sot)
    {
        task->compute();
        task->print();
    }

    // Update is used outside the solver, where there is the notion of the specific task requirements.
    sot[1] = std::make_unique<Task0>(3);
    dynamic_cast<Task1 *>(sot[0].get())->update(2, 2.71);

    for (const auto &task : sot)
    {
        task->compute();
        task->print();
    }

    return 0;
}
