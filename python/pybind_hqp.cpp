#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "hqp/hqp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyhqp, m) {
    m.doc() = "Python bindings for HQP (Hierarchical Quadratic Programming) solver";

    using HQPDynamic = hqp::HierarchicalQP<>;

    // --- HierarchicalQP solver ---
    py::class_<HQPDynamic>(m, "HierarchicalQP")
        .def(py::init<int, int>(),
             py::arg("m"), py::arg("n"),
             "Create solver with m constraints and n variables")

        .def("set_metric", &HQPDynamic::set_metric,
             py::arg("metric"),
             "Set the metric matrix (must be symmetric positive definite)")

        .def("set_problem",
             static_cast<void (HQPDynamic::*)(const Eigen::MatrixXd&, const Eigen::VectorXd&,
                                              const Eigen::VectorXd&, const Eigen::VectorXi&)>(
               &HQPDynamic::set_problem),
             py::arg("matrix"), py::arg("lower"), py::arg("upper"), py::arg("breaks"),
             "Set hierarchical QP problem with constraint matrix, bounds, and break points")

        .def("get_primal", &HQPDynamic::get_primal,
             "Solve and return the primal solution")

        .def("get_slack", &HQPDynamic::get_slack,
             "Return (slack_lower, slack_upper) tuple in original constraint order")

        .def("print_active_set", [](HQPDynamic& self) { self.print_active_set(); },
             "Print active set to stdout")

        .def_readwrite("tolerance", &HQPDynamic::tolerance,
                       "Convergence tolerance");

    // --- One-shot solve convenience function ---
    m.def(
      "solve",
      [](const Eigen::MatrixXd& matrix,
         const Eigen::VectorXd& lower,
         const Eigen::VectorXd& upper,
         const Eigen::VectorXi& breaks,
         py::object metric) -> Eigen::VectorXd {
          HQPDynamic solver(matrix.rows(), matrix.cols());
          if (!metric.is_none()) {
              solver.set_metric(metric.cast<Eigen::MatrixXd>());
          }
          solver.set_problem(matrix, lower, upper, breaks);
          return Eigen::VectorXd(solver.get_primal());
      },
      py::arg("matrix"), py::arg("lower"), py::arg("upper"), py::arg("breaks"),
      py::arg("metric") = py::none(),
      "One-shot solve: create solver, set problem, and return the primal solution");
}
