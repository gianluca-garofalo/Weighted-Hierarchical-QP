#ifndef _Task_
#define _Task_

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
        friend class HierarchicalQP;

    protected:
        Eigen::MatrixXd matrix_;
        Eigen::VectorXd vector_;
        Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
        bool isComputed_ = false;

        Task(const Eigen::Array<bool, Eigen::Dynamic, 1>&);
        virtual void compute() = 0;
        virtual ~Task() = default;
    };

} // namespace hqp

#endif // _Task_
