#ifndef _HierarchicalQP_
#define _HierarchicalQP_

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "task.hpp"
// #include "options.hpp"

namespace hqp
{

    class HierarchicalQP
    {

    private:
        uint col_;
        Eigen::VectorXd primal_;
        Eigen::VectorXd task_;
        Eigen::VectorXd guess_;
        Eigen::MatrixXd inverse_;
        Eigen::MatrixXd nullSpace_;
        Eigen::MatrixXd codRight_;
        uint k_ = 0;

        void solve();
        void equality_hqp();
        void inequality_hqp();
        void dual_update(uint h, const Eigen::VectorXd& tau);
        Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>&);
        Eigen::MatrixXd get_matrix(std::shared_ptr<Task>);
        Eigen::VectorXd get_vector(std::shared_ptr<Task>);

    public:
        double tolerance = 1e-9;
        std::vector<std::shared_ptr<Task>> sot;

        HierarchicalQP(uint n);
        Eigen::VectorXd get_primal();
        void print_active_set();
    };

} // namespace hqp

#endif // _HierarchicalQP_
