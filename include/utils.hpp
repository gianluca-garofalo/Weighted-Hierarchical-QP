/**
 * @file utils.hpp
 * @brief Declares common utility functions used throughout the HQP solver.
 *
 * The functions declared in this file assist with array and index manipulation,
 * which are critical for processing constraints and handling active sets.
 */
#ifndef _UtilsHQP_
#define _UtilsHQP_

#include <memory>
#include <type_traits>
#include <Eigen/Dense>

namespace hqp {

/**
 * @brief Identifies and returns the indices of all true elements in a given boolean array.
 *
 * Iterates over the array and collects indices where the condition evaluates to true.
 *
 * @param in Boolean array to be scanned.
 * @return VectorXi containing the positions of true values.
 */
Eigen::VectorXi find(const Eigen::Array<bool, Eigen::Dynamic, 1>&);

}  // namespace hqp

#endif  // _UtilsHQP_
