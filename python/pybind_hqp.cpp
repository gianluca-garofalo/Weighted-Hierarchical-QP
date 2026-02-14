#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include "hqp/hqp.hpp"
#include "task/task.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyhqp, m) {
    m.doc() = "Python bindings for HQP (Hierarchical Quadratic Programming) solver";
    
    // Expose the main HierarchicalQP class using the default template parameters (dynamic sizing)
    using HQPDynamic = hqp::HierarchicalQP<>;
    py::class_<HQPDynamic>(m, "HierarchicalQP")
        .def(py::init<int, int>(), 
             "Constructor for HierarchicalQP solver",
             py::arg("m"), py::arg("n"))
        
        .def("set_metric", &HQPDynamic::set_metric,
             "Set the metric matrix used to define the quadratic cost",
             py::arg("metric"))
        
        .def("set_problem", static_cast<void (HQPDynamic::*)(const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXi&)>(&HQPDynamic::set_problem),
             "Set up the hierarchical QP problem with constraint matrix, bounds, and break points",
             py::arg("matrix"), py::arg("lower"), py::arg("upper"), py::arg("breaks"))
        
        .def("get_primal", &HQPDynamic::get_primal,
             "Compute and retrieve the primal solution")
        
        .def("print_active_set", [](HQPDynamic& self) { self.print_active_set(); },
             "Print the active set details to console")
        
        .def_readwrite("tolerance", &HQPDynamic::tolerance,
                      "Tolerance for convergence and numerical stability");

}
