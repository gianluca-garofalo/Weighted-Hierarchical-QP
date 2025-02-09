#ifndef _HierarchicalQP_
#define _HierarchicalQP_

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "task.hpp"

namespace hqp
{

    class HierarchicalQP
    {

    private:
        std::vector<std::shared_ptr<Task>> sot_;
        Eigen::VectorXd primal_;
        Eigen::VectorXd tasks_;
        Eigen::VectorXd slack_;
        uint col_;
        uint k_;
        Eigen::MatrixXd nullSpace;
        Eigen::MatrixXd inverse_;
        Eigen::MatrixXd codRight;
        bool is_solved_ = false;

        void eHQP();

        Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>&);

    public:
        HierarchicalQP(uint n);
        void solve();
        void push_back(std::shared_ptr<Task> task);
        Eigen::VectorXd get_primal();
    };

} // namespace hqp

#endif // _HierarchicalQP_
