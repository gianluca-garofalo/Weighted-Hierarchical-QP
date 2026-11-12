#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>

int main() {
    // All rows are parallel to x1 + x2. Projected onto what level 0 leaves free,
    // the row of level 1 is rounding noise and must get rank 0, or the solution
    // blows up along (1, -1).
    //   Level 0: x1 + x2 <= -0.5 and x1 + x2 >= 0.5 (conflicting, best x1 + x2 = 0)
    //   Level 1: x1 + x2 <= -1                       (no freedom left for it)
    //   Level 2: x == 0
    // Solution: (0, 0)
    Eigen::MatrixXd A(6, 2);
    A << -1, -1,
          2,  2,
         -2, -2,
          2,  2,
          1,  0,
          0,  1;
    Eigen::VectorXd lower(6), upper(6);
    lower << -3, -4, -3, -4, 0, 0;
    upper <<  4, -1, -1, -2, 0, 0;
    Eigen::VectorXi breaks(3);
    breaks << 3, 4, 6;
    Eigen::Vector2d expected(0, 0);

    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_problem(A, lower, upper, breaks);
    Eigen::VectorXd x = solver.get_primal();
    std::cout << "Solution: " << x.transpose() << "  (changes=" << solver.changes << ")" << std::endl;

    if ((x - expected).norm() > 1e-9) {
        std::cerr << "Expected " << expected.transpose() << std::endl;
        return 1;
    }
    return 0;
}
