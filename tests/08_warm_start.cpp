#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>

int main() {
    // Problem with active bounds at the solution:
    //   Level 0: -1 <= x_i <= 1       (box constraints, 3 rows)
    //   Level 1: x1 + x2 + x3 <= 1    (sum constraint)
    //   Level 2: x1 - x2 == 0.5       (equality)
    //   Level 3: 10 <= 3x1+x2-x3 <= 20
    // Solution: (1, 0.5, -1) — x1 and x3 bounds are active
    Eigen::MatrixXd A(6, 3);
    A << Eigen::MatrixXd::Identity(3, 3),
         1, 1, 1,
         1, -1, 0,
         3, 1, -1;
    Eigen::VectorXd lower(6), upper(6);
    lower << -1, -1, -1, -1e9, 0.5, 10;
    upper <<  1,  1,  1,    1, 0.5, 20;
    Eigen::VectorXi breaks(4);
    breaks << 3, 4, 5, 6;

    Eigen::Vector3d expected(1, 0.5, -1);

    // 1. Cold start
    hqp::HierarchicalQP solver(A.rows(), A.cols());
    solver.set_problem(A, lower, upper, breaks);
    Eigen::VectorXd x1 = solver.get_primal();
    int changes_cold = solver.changes;
    std::cout << "Cold start:  " << x1.transpose()
              << "  (changes=" << changes_cold << ")" << std::endl;

    if (!x1.isApprox(expected)) {
        std::cerr << "Cold start wrong answer" << std::endl;
        return 1;
    }

    // 2. Warm start: guess_ = x1, re-set same problem
    solver.set_problem(A, lower, upper, breaks);
    Eigen::VectorXd x2 = solver.get_primal();
    int changes_warm = solver.changes;
    std::cout << "Warm start:  " << x2.transpose()
              << "  (changes=" << changes_warm << ")" << std::endl;

    if (!x2.isApprox(expected)) {
        std::cerr << "Warm start wrong answer" << std::endl;
        return 1;
    }

    if (changes_warm != 0) {
        std::cerr << "Warm start should need 0 active-set changes, got "
                  << changes_warm << std::endl;
        return 1;
    }

    // 3. Verify solver adapts to a new problem (not stuck on old guess)
    //    Change equality: x1 - x2 == -0.5
    lower(4) = -0.5;
    upper(4) = -0.5;

    // Use a fresh solver for the reference answer
    hqp::HierarchicalQP fresh(A.rows(), A.cols());
    fresh.set_problem(A, lower, upper, breaks);
    Eigen::VectorXd x_ref = fresh.get_primal();

    solver.set_problem(A, lower, upper, breaks);
    Eigen::VectorXd x3 = solver.get_primal();
    std::cout << "New problem: " << x3.transpose()
              << "  (changes=" << solver.changes << ")" << std::endl;

    if (!x3.isApprox(x_ref)) {
        std::cerr << "Changed problem: warm-started solver disagrees with fresh solver" << std::endl;
        std::cerr << "  Fresh:  " << x_ref.transpose() << std::endl;
        std::cerr << "  Warm:   " << x3.transpose() << std::endl;
        return 1;
    }

    std::cout << "All warm start tests passed" << std::endl;
    return 0;
}
