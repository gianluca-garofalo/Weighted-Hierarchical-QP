#include <iostream>
#include <Eigen/Dense>
#include <daqp.hpp>
#include <hqp.hpp>
#include "lexls_interface.hpp"

int main() {
    int cols = 8;
    int rows = 24;
    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd bu(rows), bl(rows);
    Eigen::VectorXi break_points = (Eigen::VectorXi(7) << 3, 5, 9, 11, 14, 16, 24).finished();

    A << 0.218964, -0.464995, 0.616472, -0.208449, -0.10418, 0.512285, 0.0282852, 0.195593, -0.485341, -0.38673, -0.248683, 0.218816, -0.341358, -0.464501, 0.33003, 0.252894, 0.382382, -0.228654, -0.435043, -0.496729, -0.258558, 0.0601361, 0.434128, 0.326442, 0.144145, -0.352011, -0.0196586, -0.65623, -0.222896, 0.383106, 0.0466607, 0.475034, -0.4534, 0.36453, 0.335822, 0.420302, -0.103684, -0.3265, 0.253642, 0.436382, -0.240293, 0.156987, -0.196622, 0.50917, -0.1916, -0.462314, 0.185311, -0.578719, 0.265133, -0.399043, 0.471711, 0.300342, 0.528176, 0.223301, 0.179581, -0.310918, -0.357731, -0.450624, -0.00787481, 0.0745105, 0.16884, -0.407995, -0.49171, -0.476032, 0.145357, -0.195635, -0.340882, -0.419088, 0.417486, -0.444923, -0.384478, -0.358732, 0.310027, -0.582462, -0.101166, 0.594892, 0.106081, 0.433189, 0.0396408, 0.00356504, -0.309173, -0.204689, -0.177491, -0.127959, 0.241252, -0.083299, 0.681861, 0.533443, -0.581285, 0.54031, 0.312086, 0.22642, 0.380468, 0.0483584, -0.227616, -0.150356, 0.015998, 0.503133, 0.557385, -0.271056, -0.312635, -0.166569, -0.296016, -0.386446, 0.160767, -0.457968, 0.0922015, -0.286843, 0.352205, -0.407232, -0.276392, -0.554401, 0.613591, 0.273318, -0.455293, 0.191626, -0.0920827, 0.0243434, 0.529794, 0.122629, -0.141638, -0.196155, 0.570321, -0.103815, 0.422384, -0.604407, -0.205398, 0.139691, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    bu << -0.236308, -1.76383, -0.0275666, 1.65649, 0.0170617, 0.186263, 0.72691, -0.00906073, -0.618032, -0.268404, -0.51175, 0.475369, 0.694832, 1.24317, 0.402622, 0.799997, 0, 0, 0, 0, 0, 0, 0, 0;
    bl << -0.236308, -1.76383, -0.0526374, 0.798752, -0.286752, -0.394221, 0.109913, -0.883363, -0.618032, -0.268404, -0.51175, 0.475369, 0.694832, 1.1731, 0.402622, -1.42504, 0, 0, 0, 0, 0, 0, 0, 0;

    auto result = daqp_solve(A, bu, bl, (Eigen::VectorXi(1 + break_points.size()) << 0, break_points).finished());
    auto daqp   = result.get_primal();
    std::cout << "DAQP solution: " << daqp.transpose() << std::endl;

    hqp::HierarchicalQP solver(rows, cols);
    solver.set_problem(A, bl, bu, break_points);
    auto hqp = solver.get_primal();
    std::cout << "HQP solution: " << hqp.transpose() << std::endl;

    auto tmp = lexls_from_stack(A, bu, bl, break_points);
    tmp.solve();
    auto lexls = tmp.get_x();
    std::cout << "LexLS solution: " << lexls.transpose() << std::endl;

    std::vector<Eigen::VectorXd> slacks;
    for (auto start = 0, k = 0; k < break_points.size(); ++k) {
        auto n_constraints = break_points(k) - start;
        std::cout << std::endl;

        Eigen::VectorXd slack_lw = A.middleRows(start, n_constraints) * daqp - bl.segment(start, n_constraints);
        Eigen::VectorXd slack_up = A.middleRows(start, n_constraints) * daqp - bu.segment(start, n_constraints);
        slack_lw = (slack_lw.array() < 0).cast<double>() * slack_lw.array();
        slack_up = (slack_up.array() > 0).cast<double>() * slack_up.array();
        std::cout << "DAQP slacks for level " << k << " (low | up) : " << slack_lw.norm() << " | " << slack_up.norm()
                  << std::endl;

        slack_lw = A.middleRows(start, n_constraints) * lexls - bl.segment(start, n_constraints);
        slack_up = A.middleRows(start, n_constraints) * lexls - bu.segment(start, n_constraints);
        slack_lw = (slack_lw.array() < 0).cast<double>() * slack_lw.array();
        slack_up = (slack_up.array() > 0).cast<double>() * slack_up.array();
        std::cout << "LexLS slacks for level " << k << " (low | up) : " << slack_lw.norm() << " | " << slack_up.norm()
                  << std::endl;
        start = break_points(k);
    }

    return hqp.isApprox(lexls) ? 0 : 1;
}
