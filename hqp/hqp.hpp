#ifndef _HierarchicalQP_
#define _HierarchicalQP_

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
        std::vector<std::shared_ptr<Task>> sot_;
        Eigen::VectorXd primal_;
        Eigen::VectorXd task_;
        Eigen::VectorXd slack_;
        uint col_;
        uint k_;
        Eigen::MatrixXd nullSpace;
        Eigen::MatrixXd inverse_;
        Eigen::MatrixXd codRight;
        Eigen::VectorXd guess_;
        double tolerance = 1e-9;
        bool isSolved_ = false;

        void eHQP();
        void iHQP();
        void dual_update(uint h, const Eigen::VectorXd& tau);

        Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>&);

        friend Eigen::VectorXd Task::get_vector();

    public:
        HierarchicalQP(uint n);
        void solve();
        void push_back(std::shared_ptr<Task> task);
        std::vector<std::shared_ptr<Task>>& get_sot();
        Eigen::VectorXd get_primal();
    };

} // namespace hqp

#endif // _HierarchicalQP_
