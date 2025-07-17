#ifndef _HierarchicalQP_UTILITIES_TPP_
#define _HierarchicalQP_UTILITIES_TPP_

#include <iostream>

namespace hqp {

// TODO: upgrade to a logger keeping track of the active set
template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS>::print_active_set() {
    std::cout << "Active set:\n";
    for (int start = 0, k = 0; k < k_; ++k) {
        std::cout << "\tLevel " << k << ":\n";
        for (int row = start; row < breaksAct_(k); ++row) {
            std::cout << "\t\t" << lower_(row) << " < " << matrix_.row(row) << " < " << upper_(row) << "\n";
        }
        start = breaks_(k);
    }
    std::cout << std::endl;
}

}  // namespace hqp

#endif  // _HierarchicalQP_UTILITIES_TPP_
