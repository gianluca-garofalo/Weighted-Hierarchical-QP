#include <iostream>
#include <Eigen/Dense>
#include <hqp/hqp.hpp>

int main() {
    std::cout << "=== Template Matrix Size Demo ===" << std::endl;

    constexpr int cols = 8;
    constexpr int rows = 24;
    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd bu(rows), bl(rows);
    Eigen::VectorXi breaks = (Eigen::VectorXi(7) << 3, 5, 9, 11, 14, 16, 24).finished();

    A << 0.218964, -0.464995, 0.616472, -0.208449, -0.10418, 0.512285, 0.0282852, 0.195593, -0.485341, -0.38673, -0.248683, 0.218816, -0.341358, -0.464501, 0.33003, 0.252894, 0.382382, -0.228654, -0.435043, -0.496729, -0.258558, 0.0601361, 0.434128, 0.326442, 0.144145, -0.352011, -0.0196586, -0.65623, -0.222896, 0.383106, 0.0466607, 0.475034, -0.4534, 0.36453, 0.335822, 0.420302, -0.103684, -0.3265, 0.253642, 0.436382, -0.240293, 0.156987, -0.196622, 0.50917, -0.1916, -0.462314, 0.185311, -0.578719, 0.265133, -0.399043, 0.471711, 0.300342, 0.528176, 0.223301, 0.179581, -0.310918, -0.357731, -0.450624, -0.00787481, 0.0745105, 0.16884, -0.407995, -0.49171, -0.476032, 0.145357, -0.195635, -0.340882, -0.419088, 0.417486, -0.444923, -0.384478, -0.358732, 0.310027, -0.582462, -0.101166, 0.594892, 0.106081, 0.433189, 0.0396408, 0.00356504, -0.309173, -0.204689, -0.177491, -0.127959, 0.241252, -0.083299, 0.681861, 0.533443, -0.581285, 0.54031, 0.312086, 0.22642, 0.380468, 0.0483584, -0.227616, -0.150356, 0.015998, 0.503133, 0.557385, -0.271056, -0.312635, -0.166569, -0.296016, -0.386446, 0.160767, -0.457968, 0.0922015, -0.286843, 0.352205, -0.407232, -0.276392, -0.554401, 0.613591, 0.273318, -0.455293, 0.191626, -0.0920827, 0.0243434, 0.529794, 0.122629, -0.141638, -0.196155, 0.570321, -0.103815, 0.422384, -0.604407, -0.205398, 0.139691, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    bu << -0.236308, -1.76383, -0.0275666, 1.65649, 0.0170617, 0.186263, 0.72691, -0.00906073, -0.618032, -0.268404, -0.51175, 0.475369, 0.694832, 1.24317, 0.402622, 0.799997, 0, 0, 0, 0, 0, 0, 0, 0;
    bl << -0.236308, -1.76383, -0.0526374, 0.798752, -0.286752, -0.394221, 0.109913, -0.883363, -0.618032, -0.268404, -0.51175, 0.475369, 0.694832, 1.1731, 0.402622, -1.42504, 0, 0, 0, 0, 0, 0, 0, 0;

    std::cout << "1. Dynamic sizing (default):" << std::endl;
    hqp::HierarchicalQP solver_dynamic(rows, cols);
    solver_dynamic.set_problem(A, bl, bu, breaks);

    std::cout << "2. Fixed maximum sizes (max 30 rows, 10 cols):" << std::endl;
    hqp::HierarchicalQP<30, 10> solver_fixed(rows, cols);
    solver_fixed.set_problem(A, bl, bu, breaks);

    std::cout << "3. Template parameter deduction with fixed-size matrices:" << std::endl;
    hqp::HierarchicalQP hqp(A.topLeftCorner<rows, cols>().eval());
    hqp.set_problem(
      A.topLeftCorner<rows, cols>().eval(), bl.head<rows>().eval(), bu.head<rows>().eval(), breaks.head<7>().eval());

    Eigen::VectorXd solution_dynamic  = solver_dynamic.get_primal();
    Eigen::VectorXd solution_fixed    = solver_fixed.get_primal();
    Eigen::VectorXd solution_template = hqp.get_primal();

    std::cout << "Dynamic solver solution: " << solution_dynamic.transpose() << std::endl;
    std::cout << "Fixed solver solution:   " << solution_fixed.transpose() << std::endl;
    std::cout << "Template solver solution: " << solution_template.transpose() << std::endl;

    bool test1 = solution_dynamic.isApprox(solution_fixed);
    bool test2 = solution_dynamic.isApprox(solution_template);

    if (test1 && test2) {
        std::cout << "✓ All solvers produce the same result!" << std::endl;
    } else {
        std::cout << "✗ Solutions differ!" << std::endl;
        if (!test1) {
            std::cout << "  Dynamic vs Fixed differ" << std::endl;
        }
        if (!test2) {
            std::cout << "  Dynamic vs Template differ" << std::endl;
        }
    }

    return (test1 && test2) ? 0 : 1;
}
