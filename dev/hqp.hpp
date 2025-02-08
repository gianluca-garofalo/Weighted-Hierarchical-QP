#ifndef _HQP_
#define _HQP_

#include <Eigen/Dense>

namespace hqp
{

    class HQP
    {

    private:
        int dof_;
        int k_ = 0;
        int numTasks_ = 0;
        // Eigen::Matrix<double, n, n> E_;
        Eigen::ArrayXi idx_;
        Eigen::ArrayXi starts_;
        Eigen::ArrayXi stops_;
        // Eigen::Array<int, 1, 2> lockedSet_;
        // Eigen::Array<int, 1, 2> activeSet_;
        Eigen::MatrixXd Matrix_;
        // Eigen::Vector<double, m> vector_;
        // Eigen::Vector<double, m> dimensions_;
        // Eigen::Vector<double, m> isEquality_;
        // Eigen::Matrix<double, m, n> Weight_;
        // Eigen::Matrix<double, m, n> Soft_;
        // Eigen::Vector<double, m> guess_;
        // Eigen::Vector<double, n> primal_;
        // Eigen::Vector<double, n> dual_;
        // Eigen::Vector<double, n> slack_;

        Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1> &in);


    public:
        HQP(int rows, int cols) : dof_{cols}, Matrix_(Eigen::MatrixXd::Zero(rows, cols)) {}

        bool insert(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXi &flag)
        {
            if (A.rows() != b.rows())
            {
                return false;
            }
            numTasks_++;
            Matrix_.middleRows(k_, A.rows()) = A;
            k_ += A.rows();
            return true;
        }

        Eigen::ArrayXi data(const Eigen::VectorXi &tasks)
        {
            Eigen::Array<bool, Eigen::Dynamic, 1> row = Eigen::ArrayXi::Zero(stops_(Eigen::last), 1).cast<bool>();
            for (auto k : tasks)
            {
                if (numTasks_ - 1 < k || k < 0)
                    throw std::runtime_error("HQP: out of bounds.");
                else
                    row = row || (starts_(k) <= idx_ && idx_ <= stops_(k));
            }
            return idx_(find(row));
        }

        Eigen::MatrixXd get_A() { return Matrix_; }
    };

} // namespace hqp

#endif // _HQP_
