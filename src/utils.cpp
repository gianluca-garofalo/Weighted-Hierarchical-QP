#include "utils.hpp"

namespace hqp {

Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>& in) {
    /**
     * @brief Identifies and returns the indices of all true elements in a given boolean array.
     *
     * Iterates over the array and collects indices where the condition evaluates to true.
     *
     * @param in Boolean array to be scanned.
     * @return VectorXi containing the positions of true values.
     */
    Eigen::VectorXi out = Eigen::VectorXi::Zero(in.cast<int>().sum());
    for (auto j = 0, i = 0; i < in.rows(); ++i) {
        if (in(i)) out(j++) = i;
    }
    return out;
}

}  // namespace hqp
