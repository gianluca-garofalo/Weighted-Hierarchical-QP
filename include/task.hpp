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
        Eigen::LLT<Eigen::MatrixXd> weight_;
        friend class HierarchicalQP;

    protected:
        Eigen::MatrixXd matrix_;
        Eigen::VectorXd vector_;
        Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
        Eigen::VectorXi indices_;
        bool isComputed_ = false;

        Task(const Eigen::Array<bool, Eigen::Dynamic, 1>&);
        virtual void compute() = 0;
        virtual ~Task() = default;

    public:
        double tolerance = 1e-9;
        
        void set_weight(const Eigen::MatrixXd&);
        void select_variables(const Eigen::VectorXi& indices);
    };

} // namespace hqp

#endif // _Task_
