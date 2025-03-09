#ifndef _Task_
#define _Task_

#include <tuple>
#include <memory>
#include <Eigen/Dense>

namespace hqp
{

    class Task
    {
    private:
        uint rank_;
        Eigen::VectorXd slack_;
        Eigen::VectorXd dual_;
        Eigen::Array<bool, Eigen::Dynamic, 1> activeSet_;
        Eigen::Array<bool, Eigen::Dynamic, 1> lockedSet_;
        Eigen::Array<bool, Eigen::Dynamic, 1> workSet_;
        Eigen::MatrixXd codMid_;
        Eigen::MatrixXd codLeft_;
        Eigen::LLT<Eigen::MatrixXd> weight_;
        friend class HierarchicalQP;
        friend class SubTasks;

    protected:
        Eigen::MatrixXd matrix_;
        Eigen::VectorXd vector_;
        Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
        Eigen::VectorXi indices_;
        bool isComputed_ = false;

        virtual void compute() = 0;

    public:
        double tolerance = 1e-9;

        Task(const Eigen::Array<bool, Eigen::Dynamic, 1>& set);
        virtual ~Task() = default;
        void select_variables(const Eigen::VectorXi& indices);
    };


    template <typename... Args>
    class TaskInterface : public Task
    {
    protected:
        std::tuple<Args...> args_;
        virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd> run(Args... args) = 0;

        void compute() override
        {
            std::tie(matrix_, vector_) = std::apply([this](auto&... unpacked) { return run(unpacked...); }, args_);
            assert(matrix_.rows() == vector_.rows());
            assert(equalitySet_.size() == vector_.rows());
            if (!indices_.size())
            {
                auto n = matrix_.cols();
                indices_ = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
            }
            isComputed_ = true;
        }

    public:
        TaskInterface(const Eigen::Array<bool, Eigen::Dynamic, 1>& set) : Task(set) {}
        void update(Args... args)
        {
            args_ = std::make_tuple(args...);
            isComputed_ = false;
        }
    };


    class SubTasks : public Task
    {
    private:
        Eigen::LLT<Eigen::MatrixXd> weight_;

    public:
        std::vector<std::unique_ptr<Task>> sot;

        SubTasks(const Eigen::Array<bool, Eigen::Dynamic, 1>& set);
        void compute() override;
        void set_weight(const Eigen::MatrixXd&);

        template <typename T>
        T* cast(uint k)
        {
            isComputed_ = false;
            return static_cast<T*>(sot[k].get());
        }
    };

} // namespace hqp

#endif // _Task_
