#ifndef HQP_CONSTRAINTS_TPP
#define HQP_CONSTRAINTS_TPP

namespace hqp {

template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::activate_constraint(int row, bool isLowerBound) {
    if (isLowerBound) {
        activeLowSet_(row) = true;
    } else {
        activeUpSet_(row) = true;
    }

    if (breaksAct_(level_(row)) < breaks_(level_(row))) {
        swap_constraints(breaksAct_(level_(row)), row);
    } else {
        throw std::runtime_error("Cannot activate more constraints than the available ones.");
    }
    ++breaksAct_(level_(row));
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::deactivate_constraint(int row) {
    activeLowSet_(row) = false;
    activeUpSet_(row)  = false;

    swap_constraints(--breaksAct_(level_(row)), row);
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::lock_constraint(int row) {
    if (breaksFix_(level_(row)) < breaksAct_(level_(row))) {
        swap_constraints(breaksFix_(level_(row)), row);
    } else {
        throw std::runtime_error("Cannot lock more constraints than the active ones.");
    }
    ++breaksFix_(level_(row));
}


template<int MaxRows, int MaxCols, int MaxLevels, int ROWS, int COLS, int LEVS>
void HierarchicalQP<MaxRows, MaxCols, MaxLevels, ROWS, COLS, LEVS>::swap_constraints(int i, int j) {
    if (i == j) {
        return;
    }

    std::swap(activeLowSet_(i), activeLowSet_(j));
    std::swap(activeUpSet_(i), activeUpSet_(j));
    std::swap(equalitySet_(i), equalitySet_(j));
    std::swap(lower_(i), lower_(j));
    std::swap(upper_(i), upper_(j));
    std::swap(dual_(i), dual_(j));
    std::swap(perm_(i), perm_(j));
    matrix_.row(i).swap(matrix_.row(j));
    codLefts_.row(i).swap(codLefts_.row(j));
}

}  // namespace hqp

#endif  // HQP_CONSTRAINTS_TPP