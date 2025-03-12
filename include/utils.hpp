/**
 * @file utils.hpp
 * @brief Declares common utility functions used throughout the HQP solver.
 *
 * The functions declared in this file assist with array and index manipulation,
 * which are critical for processing constraints and handling active sets.
 */
#ifndef _UtilsHQP_
#define _UtilsHQP_

#include <Eigen/Dense>

namespace hqp {

Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>&);

}  // namespace hqp

#endif  // _UtilsHQP_
