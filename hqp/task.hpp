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
        Eigen::VectorXd dual_;
        Eigen::Array<bool, Eigen::Dynamic, 1> is_locked_;
        Eigen::Array<bool, Eigen::Dynamic, 1> is_free_;
        uint rank_;
        Eigen::MatrixXd codMid_;
        Eigen::MatrixXd codLeft_;

        Task() {}
        virtual void compute() = 0;
        Eigen::MatrixXd get_matrix();
        Eigen::VectorXd get_vector();
        // TODO: add set_equality and other methods.
        bool set_equality(const Eigen::Array<bool, Eigen::Dynamic, 1> &);
        // bool is_active() {return is_active_.any();};
        // bool is_active(uint row) {return is_active_(row);};
    };

} // namespace hqp

#endif // _Task_
