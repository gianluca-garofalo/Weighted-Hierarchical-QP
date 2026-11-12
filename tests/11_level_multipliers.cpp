#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>

int main() {
    // The multipliers of a level must use that level's own basis even when lower
    // levels still have free directions, or level 1 cannot release the bound of
    // level 0 that holds it back.
    //   Level 0: 0.5 <= x2 <= 1.5 (three rows)
    //   Level 1: 3 <= x2 <= 4      (cannot be met, x2 = 1.5 is best)
    //   Level 2: x == 0
    // Solution: (0, 1.5)
    Eigen::MatrixXd A(6, 2);
    A << 0, -2,
         0,  2,
         0,  1,
         0,  1,
         1,  0,
         0,  1;
    Eigen::VectorXd lower(6), upper(6);
    lower << -4, -2, -3, 3, 0, 0;
    upper << -1,  3,  2, 4, 0, 0;
    Eigen::VectorXi breaks(3);
    breaks << 3, 4, 6;
    Eigen::Vector2d expected(0, 1.5);

    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_problem(A, lower, upper, breaks);
    Eigen::VectorXd x = solver.get_primal();
    std::cout << "Solution: " << x.transpose() << "  (changes=" << solver.changes << ")" << std::endl;

    if (!x.isApprox(expected)) {
        std::cerr << "Expected " << expected.transpose() << std::endl;
        return 1;
    }
    return 0;
}
