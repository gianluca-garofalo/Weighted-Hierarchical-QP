#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "hqp/hqp.hpp"
#include "task/task.hpp"

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

    // --- Task data holder ---
    py::class_<hqp::TaskBase, std::shared_ptr<hqp::TaskBase>>(m, "Task")
        .def(py::init([](const Eigen::MatrixXd& matrix,
                         const Eigen::VectorXd& lower,
                         const Eigen::VectorXd& upper) {
                 auto task    = std::make_shared<hqp::TaskBase>();
                 task->matrix = matrix;
                 task->lower  = lower;
                 task->upper  = upper;
                 return task;
             }),
             py::arg("matrix"), py::arg("lower"), py::arg("upper"),
             "Create a task with constraint matrix and bounds")
        .def_readwrite("matrix", &hqp::TaskBase::matrix, "Constraint matrix (m x n)")
        .def_readwrite("lower", &hqp::TaskBase::lower, "Lower bounds (m)")
        .def_readwrite("upper", &hqp::TaskBase::upper, "Upper bounds (m)")
        .def("__repr__", [](const hqp::TaskBase& t) {
            return "<Task " + std::to_string(t.matrix.rows()) + "x" +
                   std::to_string(t.matrix.cols()) + ">";
        });

    // --- StackOfTasks for building hierarchical problems ---
    py::class_<hqp::StackOfTasks>(m, "StackOfTasks")
        .def(py::init<>(), "Create empty stack of tasks")

        .def("add",
             [](hqp::StackOfTasks& self,
                const Eigen::MatrixXd& matrix,
                const Eigen::VectorXd& lower,
                const Eigen::VectorXd& upper) -> hqp::StackOfTasks& {
                 auto task = hqp::bind_task<>([=]() {
                     return std::make_tuple(matrix, lower, upper);
                 });
                 task->compute();
                 self.push_back(std::move(task));
                 return self;
             },
             py::arg("matrix"), py::arg("lower"), py::arg("upper"),
             py::return_value_policy::reference_internal,
             "Add a task level (returns self for chaining)")

        .def("get_stack", &hqp::StackOfTasks::get_stack,
             "Get concatenated (matrix, lower, upper, breaks) tuple")

        .def("__len__", &hqp::StackOfTasks::size)

        .def("__getitem__",
             [](hqp::StackOfTasks& self, int i) -> hqp::TaskBase& {
                 if (i < 0 || i >= self.size()) {
                     throw py::index_error("StackOfTasks index out of range");
                 }
                 return *self[i];
             },
             py::return_value_policy::reference_internal)

        .def("__repr__", [](const hqp::StackOfTasks& self) {
            return "<StackOfTasks with " + std::to_string(self.size()) + " levels>";
        });

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
