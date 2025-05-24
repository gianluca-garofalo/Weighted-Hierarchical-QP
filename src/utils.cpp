#include "utils.hpp"

namespace hqp {

Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>& in) {
    Eigen::VectorXi out = Eigen::VectorXi::Zero(in.cast<int>().sum());
    for (auto j = 0, i = 0; i < in.rows(); ++i) {
        if (in(i)) out(j++) = i;
    }
    return out;
}

}  // namespace hqp
