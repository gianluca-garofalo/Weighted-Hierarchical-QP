#ifndef _HQP_
#define _HQP_

#include <Eigen/Dense>

namespace hqp
{

    template <int m, int n>
    class HQP
    {

    private:
        Eigen::Matrix<double, m, n> Matrix_;

    public:
        HQP()
        {
            Matrix_ = Eigen::Matrix<double, m, n>::Identity();
        }

        HQP(const Eigen::Matrix<double, m, n> &A)
        {
            Matrix_ = A;
        }

        Eigen::Matrix<double, m, n> get_A() { return Matrix_; };
    };

    Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1> &in)
    {
        Eigen::VectorXi out = Eigen::VectorXi::Zero(in.cast<int>().sum());
        for (auto j = 0, i = 0; i < in.rows(); ++i)
        {
            if (in(i))
            {
                out(j++) = i;
            }
        }
        return out;
    }

}

#endif // _HQP_
