#include <pybind11/pybind11.h>

#include "pygadjoints/py_elasticity.hpp"

namespace py = pybind11;

void add_adjoint_class(py::module_& m) {
  py::class_<test::printer::Printer> klasse(m, "Printer");
  klasse.def(py::init<>())
      .def("print", &test::printer::Printer::print, py::arg("int"));
}
