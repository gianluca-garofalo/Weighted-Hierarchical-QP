#ifndef _HierarchicalQP_
#define _HierarchicalQP_

#include <vector>
#include <tuple>
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
        Eigen::MatrixXd cholMetric_;
        uint k_ = 0;

        void solve();
        void equality_hqp();
        void inequality_hqp();
        void dual_update(uint h, const Eigen::VectorXd& tau);
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> get_task(std::shared_ptr<Task> task, const Eigen::VectorXi& row);

    public:
        double tolerance = 1e-9;
        std::vector<std::shared_ptr<Task>> sot;

        HierarchicalQP(uint n);
        void set_metric(const Eigen::MatrixXd&);
        Eigen::VectorXd get_primal();
        void print_active_set();
    };

} // namespace hqp

#endif // _HierarchicalQP_
