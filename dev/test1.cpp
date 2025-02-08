// g++ test1.cpp -I/usr/include/eigen3 -o test1
// ./test1

#include <iostream>
#include <Eigen/Dense>

#include <functional>

// Example function to be called
void exampleFunction(int a, double b, const std::string &c)
{
    std::cout << "Integer: " << a << ", Double: " << b << ", String: " << c << std::endl;
}

void Task1(int a, std::string c)
{
    std::cout << "Integer: " << a << ", String: " << c << std::endl;
}

template <typename... Args>
struct TaskFunction
{
    std::function<void(Args...)> func_;
    std::tuple<Args...> args_;
    void compute() { std::apply(func_, args_); }
};

template <typename... Args>
using Functions = std::function<void(Args...)>;

template <typename... Args>
using Map = std::pair<std::function<void(Args...)>, std::tuple<Args...>>;

// template <typename... Args>
// using Maps = std::tuple<Map<Args...>>;

int main()
{
    Eigen::Array<int, 1, 2> starts_{0, 3};
    Eigen::Array<int, 1, 2> stops_{2, 6};
    Eigen::Array<int, 10, 1> idx_{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Eigen::Array<bool, 10, 1> rows = Eigen::Array<int, 10, 1>::Zero().cast<bool>();

    for (auto i = 0; i < starts_.size(); ++i)
    {
        rows = rows || ((starts_(i) <= idx_) && (idx_ <= stops_(i)));
    }

    std::cout << "idx: " << rows.transpose() << std::endl;
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(10, 10);
    // std::cout << "A:\n"
    //           << A(hqp::find(rows), Eigen::all) << std::endl;


    TaskFunction<int, double, const std::string &> tf = {exampleFunction, std::make_tuple(42, 3.14, "Hello, World!")};
    tf.compute();

    // Call the function with variadic arguments
    Functions funcs = exampleFunction;
    funcs(42, 3.14, "Awesome, World!");
    std::tuple sof{exampleFunction, Task1};
    std::get<0>(sof)(42, 3.14, "Awesome, World!");
    std::get<1>(sof)(42, "Awesome, World!");
    Map map0 = {exampleFunction, std::make_tuple(42, 3.14, "Hello, World!")};
    Map map1 = {Task1, std::make_tuple(42, "Hello, World!")};

    return 0;
}
