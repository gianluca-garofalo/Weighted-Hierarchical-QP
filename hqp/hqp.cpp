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
        iHQP();
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
        if (!is_solved_) solve();
        return primal_;
    }

    void HierarchicalQP::iHQP()
    {
        for (auto& task : sot_)
        {
            task->is_locked_.setZero();
            task->dual_.setZero();
        }
        uint h = 0;
        bool is_active_set_new = true;

        while (h < sot_.size())
        {
            while (is_active_set_new)
            {
                eHQP();

                // Add tasks to the active set.
                is_active_set_new = false;
                for (uint k = 0; k < k_; ++k)
                {
                    auto row = find(!sot_[k]->is_active_);
                    Eigen::MatrixXd matrix = sot_[k]->get_matrix();
                    Eigen::VectorXd vector = sot_[k]->get_vector();
                    // TODO: set tolerance as a parameter.
                    sot_[k]->is_active_(row) = (vector(row) - matrix(row, Eigen::all) * primal_).array() > 1e-9;
                    is_active_set_new = is_active_set_new || sot_[k]->is_active_(row).any();
                    // TODO: break if is_active_set_new.
                }
            }

            // Remove tasks from the active set.
            while (h < sot_.size())
            {
                for (uint k = 0; k <= h; ++k)
                {
                    sot_[k]->is_free_ = sot_[k]->is_active_ && !sot_[k]->is_equality_ && !sot_[k]->is_locked_;
                }

                if (sot_[h]->is_free_.any())
                {
                    auto row = find(sot_[h]->is_free_);
                    Eigen::MatrixXd matrix = sot_[h]->get_matrix();
                    Eigen::VectorXd vector = sot_[h]->get_vector();

                    if (h >= k_)
                    {
                        sot_[h]->slack_(row) = matrix(row, Eigen::all) * primal_ - vector(row);
                    }
                    sot_[h]->dual_(row) = sot_[h]->slack_(row);
                    dual_update(h, matrix(row, Eigen::all).transpose() * sot_[h]->dual_(row));

                    for (uint k = 0; k <= h; ++k)
                    {
                        if (sot_[k]->is_free_.any())
                        {
                            auto row = find(sot_[k]->is_free_);
                            auto test = (sot_[k]->dual_(row)).array() > 1e-9;
                            sot_[k]->is_active_(row) = !test;
                            sot_[k]->is_locked_(row) = test;
                            is_active_set_new = is_active_set_new || (k < k_ && test.any());
                        }
                    }
                }

                if (is_active_set_new) break;
                h++;
            }
        }
    }


    void HierarchicalQP::dual_update(uint h, const Eigen::VectorXd& tau)
    {
        uint k = 0;
        auto dof = col_;
        auto oldDof = col_;

        while (k < h && dof > 0)
        {
            if (sot_[k]->is_active_.any())
            {
                uint leftDof = dof - sot_[k]->rank_;
                auto row = find(sot_[k]->is_active_ && sot_[k]->is_free_);
                if (row.any())
                {
                    Eigen::VectorXd f = inverse_.block(0, col_ - dof, oldDof, sot_[k]->rank_).transpose() * tau.head(oldDof);
                    sot_[k]->codMid_
                        .topLeftCorner(sot_[k]->rank_, sot_[k]->rank_)
                        .triangularView<Eigen::Upper>()
                        .transpose()
                        .solveInPlace<Eigen::OnTheLeft>(f);
                    sot_[k]->dual_(row) = -sot_[k]->codLeft_(row, Eigen::seqN(0, sot_[k]->rank_)) * f;
                }

                oldDof = dof;
                dof = leftDof;
            }
            k++;
        }

        while (k < h)
        {
            auto row = find(sot_[k]->is_free_);
            sot_[k]->dual_(row).setZero();
            k++;
        }
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
