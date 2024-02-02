#include "pygadjoints/py_poisson.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void add_adjoint_class(py::module_ &m) {
  py::class_<pygadjoints::PoissonProblem> klasse(m, "PoissonProblem");
  klasse
      .def(py::init<>())
      // Manipulating the problem and initialization
      .def("init", &pygadjoints::PoissonProblem::Init, py::arg("fname"),
           py::arg("refinements"))

      // IO routines
      .def("export_paraview", &pygadjoints::PoissonProblem::ExportParaview,
           py::arg("fname"), py::arg("plot_elements"), py::arg("sample_rate"),
           py::arg("binary"))
      .def("export_xml", &pygadjoints::PoissonProblem::ExportXML,
           py::arg("fname"))

      // Assembly and Solving the linear problem
      .def("assemble", &pygadjoints::PoissonProblem::Assemble)
      .def("solve_linear_system",
           &pygadjoints::PoissonProblem::SolveLinearSystem)

  // OpenMP specifics
#ifdef PYGADJOINTS_USE_OPENMP
      .def("set_number_of_threads",
           &pygadjoints::PoissonProblem::SetNumberOfThreads,
           py::arg("nthreads"))
#endif
      ;
}
