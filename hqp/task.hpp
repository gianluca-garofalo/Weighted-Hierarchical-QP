#ifndef _Task_
#define _Task_

#include <Eigen/Dense>

namespace hqp
{

    class Task
    {
    private:
        bool set_guess(const Eigen::VectorXd&);
        Eigen::VectorXd guess_;
        bool hasGuess_ = false;
        friend class HierarchicalQP;

    protected:
        Eigen::MatrixXd matrix_;
        Eigen::VectorXd vector_;
        bool isComputed_ = false;

    public:
        Eigen::Array<bool, Eigen::Dynamic, 1> equalitySet_;
        Eigen::Array<bool, Eigen::Dynamic, 1> activeSet_;
        Eigen::VectorXd slack_;
        Eigen::VectorXd dual_;
        Eigen::Array<bool, Eigen::Dynamic, 1> lockedSet_;
        Eigen::Array<bool, Eigen::Dynamic, 1> workSet_;
        uint rank_;
        Eigen::MatrixXd codMid_;
        Eigen::MatrixXd codLeft_;

        Task() {}
        virtual void compute() = 0;
        Eigen::MatrixXd get_matrix();
        Eigen::VectorXd get_vector();
        // TODO: add set_equality and other methods.
        bool set_equality(const Eigen::Array<bool, Eigen::Dynamic, 1>&);
        // bool is_active() {return activeSet_.any();};
        // bool is_active(uint row) {return activeSet_(row);};
    };

} // namespace hqp

#endif // _Task_
