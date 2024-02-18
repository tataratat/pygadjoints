#include "pygadjoints/py_diffusion.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void add_diffusion_problem(py::module_ &m) {
  py::class_<pygadjoints::DiffusionProblem> klasse(m, "DiffusionProblem");
  klasse
      .def(py::init<>())
      // Manipulating the problem and initialization
      .def("init", &pygadjoints::DiffusionProblem::Init, py::arg("fname"),
           py::arg("refinements"), py::arg("print_summary") = false)
      .def("set_material_constants",
           &pygadjoints::DiffusionProblem::SetMaterialConstants,
           py::arg("thermal_diffusivity"))

      // IO routines
      .def("read_control_point_sensitivities",
           &pygadjoints::DiffusionProblem::GetParameterSensitivities,
           py::arg("fname"))
      .def("update_geometry", &pygadjoints::DiffusionProblem::UpdateGeometry,
           py::arg("fname"), py::arg("topology_changes"))
      .def("export_paraview", &pygadjoints::DiffusionProblem::ExportParaview,
           py::arg("fname"), py::arg("plot_elements"), py::arg("sample_rate"),
           py::arg("binary"))
      .def("export_xml", &pygadjoints::DiffusionProblem::ExportXML,
           py::arg("fname"))

      // Assembly and Solving the linear problem
      .def("assemble", &pygadjoints::DiffusionProblem::Assemble)
      .def("solve_linear_system",
           &pygadjoints::DiffusionProblem::SolveLinearSystem)

  // OpenMP specifics
#ifdef PYGADJOINTS_USE_OPENMP
      .def("set_number_of_threads",
           &pygadjoints::DiffusionProblem::SetNumberOfThreads,
           py::arg("nthreads"))
#endif
      ;
}
