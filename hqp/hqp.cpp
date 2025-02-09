#include "hqp.hpp"

namespace hqp
{

    HierarchicalQP::HierarchicalQP(uint n)
        : primal_{ Eigen::VectorXd::Zero(n) }
        , tasks_{ Eigen::VectorXd::Zero(n) }
        , col_{ n }
        , nullSpace{ Eigen::MatrixXd::Identity(n, n) }
        , inverse_{ Eigen::MatrixXd::Zero(n, n) }
        , codRight{ Eigen::MatrixXd::Zero(n, n) }
    {
    }

    void HierarchicalQP::solve()
    {
        for (auto& task : sot_)
        {
            task->is_active_ = task->is_equality_;
        }
        eHQP();
        is_solved_ = true;
    }

    void HierarchicalQP::eHQP()
    {
        primal_.setZero();
        auto dof = col_;
        k_ = 0;
        // TODO: replace identity with Cholesky factor.
        nullSpace.setIdentity();
        while (k_ < sot_.size() && dof > 0)
        {
            if (sot_[k_]->is_active_.any())
            {
                auto row = find(sot_[k_]->is_active_);
                Eigen::MatrixXd matrix = sot_[k_]->get_matrix()(row, Eigen::all);
                Eigen::VectorXd vector = sot_[k_]->get_vector()(row) - matrix * primal_;
                assert(matrix.cols() == col_);

                Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(matrix * nullSpace.leftCols(dof));
                // TODO: each task should have its own threshold.
                cod.setThreshold(1e-3);
                auto rank = cod.rank();
                auto leftDof = dof - rank;
                if (leftDof > 0)
                {
                    codRight.leftCols(dof) = nullSpace.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
                    nullSpace.leftCols(leftDof) = codRight.rightCols(leftDof);
                }
                else
                {
                    // In this case matrixZ() is the identity, so Eigen does not compute it explicitly and matrixZ() returns garbage
                    codRight.leftCols(dof) = nullSpace.leftCols(dof) * cod.colsPermutation();
                }
                Eigen::MatrixXd codLeft = cod.householderQ();

                inverse_.middleCols(col_ - dof, rank) = codRight.leftCols(rank);
                tasks_.segment(col_ - dof, rank) = codLeft.leftCols(rank).transpose() * vector;
                sot_[k_]->slack_(row) = codLeft.leftCols(rank) * tasks_.segment(col_ - dof, rank) - vector;
                cod.matrixT()
                    .topLeftCorner(rank, rank)
                    .triangularView<Eigen::Upper>()
                    .solveInPlace<Eigen::OnTheLeft>(tasks_.segment(col_ - dof, rank));
                primal_ += inverse_.middleCols(col_ - dof, rank) * tasks_.segment(col_ - dof, rank);

                dof = leftDof;
                sot_[k_]->rank_ = rank;
                sot_[k_]->codMid_.topLeftCorner(rank, rank) = cod.matrixT().topLeftCorner(rank, rank);
                sot_[k_]->codLeft_(row, Eigen::seqN(0, rank)) = codLeft.leftCols(rank);
            }
            k_++;
        }
    }

    void HierarchicalQP::push_back(std::shared_ptr<Task> task)
    {
        sot_.push_back(task);
    }

    Eigen::VectorXd HierarchicalQP::get_primal()
    {
        if (!is_solved_)
            solve();
        return primal_;
    }


    Eigen::VectorXi HierarchicalQP::find(const Eigen::Array<bool, Eigen::Dynamic, 1>& in)
    {
        Eigen::VectorXi out = Eigen::VectorXi::Zero(in.cast<int>().sum());
        for (auto j = 0, i = 0; i < in.rows(); ++i)
        {
            if (in(i)) out(j++) = i;
        }
        return out;
    }

} // namespace hqp
