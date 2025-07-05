/**
 * @file utils.cpp
 * @brief Provides utility functions for the HQP solver.
 *
 * This module contains helper functions that perform common operations used in the solver,
 * such as identifying indices where conditions are met.
 */
#include "utils.hpp"

namespace hqp {

Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>& in) {
    Eigen::VectorXi out = Eigen::VectorXi::Zero(in.cast<int>().sum());
    for (auto j = 0, i = 0; i < in.rows(); ++i) {
        if (in(i)) {
            out(j++) = i;
        }
    }
    return out;
}

}  // namespace hqp
