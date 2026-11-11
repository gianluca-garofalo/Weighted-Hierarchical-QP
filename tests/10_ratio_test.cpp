#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>

int main() {
    // The active-set search must not step past inactive rows of levels already
    // solved (#7). Without a ratio test this problem, seed 1633 of the
    // 04_random_problem generator reduced to 6 rows, ends with level 1 cost
    // 6.42 instead of the optimum 5.806.
    //   Level 0: three two-sided inequalities, feasible together
    //   Level 1: one inequality that cannot be met under level 0
    //   Level 2: x == 0
    Eigen::MatrixXd A(6, 2);
    A << 0.94938431195970785, 0.31411690213166171,
        -0.34438211051399187, -0.9388295702404823,
         0.81036629268301097, -0.58592360567158641,
         0.69931374302526361, 0.71481486331496746,
         1, 0,
         0, 1;
    Eigen::VectorXd lower(6), upper(6);
    lower << -2.8907654755895202, -1.1453187756329379, -0.80526566693063739, 3.293845404453144, 0, 0;
    upper << 0.40657261602089423, 0.035593689003228367, -0.46677367697495692, 4.2613014038087895, 0, 0;
    Eigen::VectorXi breaks(3);
    breaks << 3, 4, 6;

    // Lexicographic optimum, from LexLS.
    Eigen::Vector2d expected(0.0280136139523878, 1.20966725392553);

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
