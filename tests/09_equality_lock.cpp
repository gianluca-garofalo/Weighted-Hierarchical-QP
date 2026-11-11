#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>

int main() {
    // An equality that comes after an inactive row in its level must still be
    // locked, or the solver drops it and adds it back on every solve (#6).
    //   Level 0: -10 <= x1 - x2 <= 10 (inactive), x1 + x2 == 1
    //   Level 1: x1 == -3, x2 == -3   (pulls against the equality)
    // Solution: (0.5, 0.5)
    Eigen::MatrixXd A(4, 2);
    A << 1, -1,
         1,  1,
         1,  0,
         0,  1;
    Eigen::VectorXd lower(4), upper(4);
    lower << -10, 1, -3, -3;
    upper <<  10, 1, -3, -3;
    Eigen::VectorXi breaks(2);
    breaks << 2, 4;
    Eigen::Vector2d expected(0.5, 0.5);

    hqp::HierarchicalQP solver(A.rows(), A.cols());
    for (const char* start : {"Cold", "Warm"}) {
        solver.set_problem(A, lower, upper, breaks);
        Eigen::VectorXd x = solver.get_primal();
        std::cout << start << " start: " << x.transpose() << "  (changes=" << solver.changes << ")" << std::endl;
        if (!x.isApprox(expected) || solver.changes != 0) {
            std::cerr << start << " start should give " << expected.transpose() << " with 0 changes" << std::endl;
            return 1;
        }
    }
    return 0;
}
