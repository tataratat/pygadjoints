#include <pybind11/pybind11.h>

// declare signatures


PYBIND11_MODULE(pygadjoints, m) {

  add_adjoint_class(m, "elas");
}
