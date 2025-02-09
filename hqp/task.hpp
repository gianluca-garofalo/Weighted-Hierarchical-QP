#ifndef _Task_
#define _Task_

#include <Eigen/Dense>

namespace hqp
{

    class Task
    {
    protected:
        Eigen::MatrixXd matrix_;
        Eigen::VectorXd vector_;
        bool is_computed_ = false;

    public:
        Eigen::Array<bool, Eigen::Dynamic, 1> is_equality_;
        Eigen::Array<bool, Eigen::Dynamic, 1> is_active_;
        Eigen::VectorXd slack_;
        uint rank_;
        Eigen::MatrixXd codMid_;
        Eigen::MatrixXd codLeft_;
        Task() {}
        virtual void compute() = 0;
        Eigen::MatrixXd get_matrix() { if (!is_computed_) compute(); return matrix_; }
        Eigen::VectorXd get_vector() { if (!is_computed_) compute(); return vector_; }
        // TODO: add set_equality and other methods.
        // bool set_equality(const Eigen::VectorXi &is_equality) { is_equality_ = is_equality; }
    };

} // namespace hqp

#endif // _Task_
