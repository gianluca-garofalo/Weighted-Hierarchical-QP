#include "hqp.hpp"

namespace hqp
{

    HierarchicalQP::HierarchicalQP(uint n)
        : primal_{ Eigen::VectorXd::Zero(n) }
        , task_{ Eigen::VectorXd::Zero(n) }
        , col_{ n }
        , nullSpace{ Eigen::MatrixXd::Identity(n, n) }
        , inverse_{ Eigen::MatrixXd::Zero(n, n) }
        , codRight{ Eigen::MatrixXd::Zero(n, n) }
        , guess_{ Eigen::VectorXd::Zero(n) }
    {
    }

    void HierarchicalQP::solve()
    {
        bool isAllEquality = true;
        for (auto& task : sot_)
        {
            if (!task->activeSet_.size()) task->activeSet_ = task->equalitySet_;
            // Shift problem to the origin
            task->set_guess(guess_);
            isAllEquality = isAllEquality && task->equalitySet_.all();
        }
        if (isAllEquality) eHQP();
        else iHQP();
        // Shift problem back
        primal_ += guess_;
        guess_ = primal_;
        isSolved_ = true;
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
            if (sot_[k_]->activeSet_.any())
            {
                auto row = find(sot_[k_]->activeSet_);
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
                task_.segment(col_ - dof, rank) = codLeft.leftCols(rank).transpose() * vector;
                sot_[k_]->slack_(row) = codLeft.leftCols(rank) * task_.segment(col_ - dof, rank) - vector;
                cod.matrixT()
                    .topLeftCorner(rank, rank)
                    .triangularView<Eigen::Upper>()
                    .solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, rank));
                primal_ += inverse_.middleCols(col_ - dof, rank) * task_.segment(col_ - dof, rank);

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

    std::vector<std::shared_ptr<Task>>& HierarchicalQP::get_sot()
    {
        return sot_;
    }

    Eigen::VectorXd HierarchicalQP::get_primal()
    {
        if (!isSolved_) solve();
        return primal_;
    }

    void HierarchicalQP::iHQP()
    {
        for (auto& task : sot_)
        {
            task->lockedSet_.setZero();
            task->dual_.setZero();
        }
        uint h = 0;
        bool isActiveSetNew = true;

        while (h < sot_.size())
        {
            while (isActiveSetNew)
            {
                eHQP();

                // Add tasks to the active set.
                isActiveSetNew = false;
                for (uint k = 0; k < k_ && !isActiveSetNew; ++k)
                {
                    auto row = find(!sot_[k]->activeSet_);
                    Eigen::MatrixXd matrix = sot_[k]->get_matrix();
                    Eigen::VectorXd vector = sot_[k]->get_vector();
                    sot_[k]->activeSet_(row) = (vector(row) - matrix(row, Eigen::all) * primal_).array() > tolerance;
                    isActiveSetNew = sot_[k]->activeSet_(row).any();
                }
            }

            // Remove tasks from the active set.
            for (uint k = 0; k <= h; ++k)
            {
                sot_[k]->workSet_ = sot_[k]->activeSet_ && !sot_[k]->equalitySet_ && !sot_[k]->lockedSet_;
            }

            if (sot_[h]->workSet_.any())
            {
                auto row = find(sot_[h]->workSet_);
                Eigen::MatrixXd matrix = sot_[h]->get_matrix();
                Eigen::VectorXd vector = sot_[h]->get_vector();

                if (h >= k_)
                {
                    sot_[h]->slack_(row) = matrix(row, Eigen::all) * primal_ - vector(row);
                }
                sot_[h]->dual_(row) = sot_[h]->slack_(row);
                dual_update(h, matrix(row, Eigen::all).transpose() * sot_[h]->dual_(row));

                for (uint k = 0; k <= h && !isActiveSetNew; ++k)
                {
                    if (sot_[k]->workSet_.any())
                    {
                        auto row = find(sot_[k]->workSet_);
                        auto test = (sot_[k]->dual_(row)).array() > tolerance;
                        sot_[k]->activeSet_(row) = !test;
                        sot_[k]->lockedSet_(row) = test;
                        isActiveSetNew = k < k_ && test.any();
                    }
                }
            }

            if (!isActiveSetNew) h++;
        }
    }


    void HierarchicalQP::dual_update(uint h, const Eigen::VectorXd& tau)
    {
        uint k = 0;
        auto dof = col_;
        auto oldDof = col_;

        while (k < h && dof > 0)
        {
            if (sot_[k]->activeSet_.any())
            {
                uint leftDof = dof - sot_[k]->rank_;
                auto row = find(sot_[k]->activeSet_ && sot_[k]->workSet_);
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
            auto row = find(sot_[k]->workSet_);
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
