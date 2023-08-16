#include "pygadjoints/py_elasticity.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void add_adjoint_class(py::module_& m) {
  py::class_<pygadjoints::LinearElasticityProblem> klasse(
      m, "LinearElasticityProblem");
  klasse.def(py::init<>())
      .def("set_material_constants",
           &pygadjoints::LinearElasticityProblem::SetMaterialConstants,
           py::arg("lame_lambda"), py::arg("lame_mu"), py::arg("mu"))
      .def("read_from_input_file",
           &pygadjoints::LinearElasticityProblem::ReadInputFromFile,
           py::arg("fname"))
      .def("init", &pygadjoints::LinearElasticityProblem::Init,
           py::arg("refinements"))
      .def("assemble", &pygadjoints::LinearElasticityProblem::Assemble)
      .def("solve_linear_system",
           &pygadjoints::LinearElasticityProblem::SolveLinearSystem)
      .def("export_paraview",
           &pygadjoints::LinearElasticityProblem::ExportParaview,
           py::arg("fname"), py::arg("plot_elements"), py::arg("sample_rate"))
      .def("export_xml", &pygadjoints::LinearElasticityProblem::ExportXML,
           py::arg("fname"))
#ifdef PYGADJOINTS_USE_OPENMP
      .def("set_number_of_threads",
           &pygadjoints::LinearElasticityProblem::SetNumberOfThreads,
           py::arg("nthreads"))
#endif
      .def("print", &pygadjoints::LinearElasticityProblem::print,
           py::arg("int"));
}
