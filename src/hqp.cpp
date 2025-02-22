#include "hqp.hpp"

namespace hqp
{

    HierarchicalQP::HierarchicalQP(uint n)
        : col_{ n }
        , primal_{ Eigen::VectorXd::Zero(n) }
        , task_{ Eigen::VectorXd::Zero(n) }
        , guess_{ Eigen::VectorXd::Zero(n) }
        , inverse_{ Eigen::MatrixXd::Zero(n, n) }
        , nullSpace_{ Eigen::MatrixXd::Identity(n, n) }
        , codRight_{ Eigen::MatrixXd::Zero(n, n) }
    {
    }

    void HierarchicalQP::solve()
    {
        bool isAllEquality = true;
        for (auto& task : sot)
        {
            if (!task->activeSet_.size()) task->activeSet_ = task->equalitySet_;
            isAllEquality = isAllEquality && task->equalitySet_.all();
        }
        if (isAllEquality) equality_hqp();
        else inequality_hqp();

        // Shift problem back
        primal_ += guess_;
        guess_ = primal_;

        // Deactivate unused tasks for next guess
        for (uint k = k_; k < sot.size(); ++k)
        {
            auto row = find(!sot[k]->equalitySet_);
            sot[k]->activeSet_(row).setZero();
        }
    }


    void HierarchicalQP::equality_hqp()
    {
        primal_.setZero();
        auto dof = col_;
        k_ = 0;
        // TODO: replace identity with Cholesky factor.
        nullSpace_.setIdentity();
        // Eigen::MatrixXd L( P.llt().matrixL() );
        while (k_ < sot.size() && dof > 0)
        {
            if (sot[k_]->activeSet_.any())
            {
                auto row = find(sot[k_]->activeSet_);
                Eigen::MatrixXd matrix = get_matrix(sot[k_])(row, Eigen::all);
                Eigen::VectorXd vector = get_vector(sot[k_])(row) - matrix * primal_;
                assert(matrix.cols() == col_);

                Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(matrix * nullSpace_.leftCols(dof));
                // TODO: each task should have its own threshold.
                cod.setThreshold(1e-3);
                auto rank = cod.rank();
                auto leftDof = dof - rank;
                if (leftDof > 0)
                {
                    codRight_.leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation() * cod.matrixZ().transpose();
                    nullSpace_.leftCols(leftDof) = codRight_.rightCols(leftDof);
                }
                else
                {
                    // In this case matrixZ() is the identity, so Eigen does not compute it explicitly and matrixZ() returns garbage
                    codRight_.leftCols(dof) = nullSpace_.leftCols(dof) * cod.colsPermutation();
                }
                Eigen::MatrixXd codLeft_ = cod.householderQ();

                inverse_.middleCols(col_ - dof, rank) = codRight_.leftCols(rank);
                task_.segment(col_ - dof, rank) = codLeft_.leftCols(rank).transpose() * vector;
                sot[k_]->slack_(row) = codLeft_.leftCols(rank) * task_.segment(col_ - dof, rank) - vector;
                cod.matrixT()
                    .topLeftCorner(rank, rank)
                    .triangularView<Eigen::Upper>()
                    .solveInPlace<Eigen::OnTheLeft>(task_.segment(col_ - dof, rank));
                primal_ += inverse_.middleCols(col_ - dof, rank) * task_.segment(col_ - dof, rank);

                dof = leftDof;
                sot[k_]->rank_ = rank;
                sot[k_]->codMid_.topLeftCorner(rank, rank) = cod.matrixT().topLeftCorner(rank, rank);
                sot[k_]->codLeft_(row, Eigen::seqN(0, rank)) = codLeft_.leftCols(rank);
            }
            k_++;
        }
    }


    Eigen::VectorXd HierarchicalQP::get_primal()
    {
        uint k = 0;
        while (k < k_ && sot[k]->isComputed_) k++;
        if (k <= k_) solve();
        return primal_;
    }

    void HierarchicalQP::inequality_hqp()
    {
        for (auto& task : sot)
        {
            task->lockedSet_.setZero();
            task->dual_.setZero();
        }
        uint h = 0;
        bool isActiveSetNew = true;

        while (h < sot.size())
        {
            while (isActiveSetNew)
            {
                equality_hqp();

                // Add tasks to the active set.
                isActiveSetNew = false;
                for (uint k = 0; k < k_ && !isActiveSetNew; ++k)
                {
                    auto row = find(!sot[k]->activeSet_);
                    Eigen::MatrixXd matrix = get_matrix(sot[k]);
                    Eigen::VectorXd vector = get_vector(sot[k]);
                    sot[k]->activeSet_(row) = (vector(row) - matrix(row, Eigen::all) * primal_).array() > tolerance;
                    isActiveSetNew = sot[k]->activeSet_(row).any();
                }
            }

            // Remove tasks from the active set.
            for (uint k = 0; k <= h; ++k)
            {
                sot[k]->workSet_ = sot[k]->activeSet_ && !sot[k]->equalitySet_ && !sot[k]->lockedSet_;
            }

            if (sot[h]->workSet_.any())
            {
                auto row = find(sot[h]->workSet_);
                Eigen::MatrixXd matrix = get_matrix(sot[h]);
                Eigen::VectorXd vector = get_vector(sot[h]);

                if (h >= k_)
                {
                    sot[h]->slack_(row) = matrix(row, Eigen::all) * primal_ - vector(row);
                }
                sot[h]->dual_(row) = sot[h]->slack_(row);
                dual_update(h, matrix(row, Eigen::all).transpose() * sot[h]->dual_(row));

                for (uint k = 0; k <= h && !isActiveSetNew; ++k)
                {
                    if (sot[k]->workSet_.any())
                    {
                        auto row = find(sot[k]->workSet_);
                        auto test = (sot[k]->dual_(row)).array() > tolerance;
                        sot[k]->activeSet_(row) = !test;
                        sot[k]->lockedSet_(row) = test;
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
            if (sot[k]->activeSet_.any())
            {
                uint leftDof = dof - sot[k]->rank_;
                auto row = find(sot[k]->activeSet_ && sot[k]->workSet_);
                if (row.any())
                {
                    Eigen::VectorXd f = inverse_.block(0, col_ - dof, oldDof, sot[k]->rank_).transpose() * tau.head(oldDof);
                    sot[k]->codMid_
                        .topLeftCorner(sot[k]->rank_, sot[k]->rank_)
                        .triangularView<Eigen::Upper>()
                        .transpose()
                        .solveInPlace<Eigen::OnTheLeft>(f);
                    sot[k]->dual_(row) = -sot[k]->codLeft_(row, Eigen::seqN(0, sot[k]->rank_)) * f;
                }

                oldDof = dof;
                dof = leftDof;
            }
            k++;
        }

        while (k < h)
        {
            auto row = find(sot[k]->workSet_);
            sot[k]->dual_(row).setZero();
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

    Eigen::MatrixXd HierarchicalQP::get_matrix(std::shared_ptr<Task> task)
    {
        if (!task->isComputed_) task->compute();
        return task->matrix_;
    }

    Eigen::VectorXd HierarchicalQP::get_vector(std::shared_ptr<Task> task)
    {
        if (!task->isComputed_)
        {
            task->compute();
            task->vector_ -= task->matrix_ * guess_; // Shift problem to the origin
        }
        return task->vector_;
    }

    void HierarchicalQP::print_active_set()
    {
        std::cout << "Active set:\n";
        for (uint k = 0; const auto & task : sot)
        {
            if (k < k_ && task->activeSet_.any())
            {
                std::cout << "\tLevel " << k << " -> constraints " << find(task->activeSet_).transpose() << "\n";
            }
            k++;
        }
        std::cout << std::endl;
    }

} // namespace hqp
