#include <gismo.h>

namespace test::printer {
class Printer {
 public:
  Printer() = default;

  print(const int i) {
    for (int j{}; j < i; i++) {
      GISMO_INFO("TEST IF GISMO IS THERE SUCCEEDED")
    };
  }
}

template <typename Type>
void add_adjoint_class(py::module& m, const char* class_name) {
  py::class_<test::printer::Printer> klasse(m, class_name);
  klasse.def(py::init<>())
      .def("print", &test::printer::print, py::arg("int");
}
}  // namespace test::printer