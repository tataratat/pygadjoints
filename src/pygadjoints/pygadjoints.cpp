#include <pybind11/pybind11.h>

// declare signatures
void add_elasticity_problem(pybind11::module_ &);
void add_diffusion_problem(pybind11::module_ &);

PYBIND11_MODULE(pygadjoints, m) {
  add_elasticity_problem(m);
  add_diffusion_problem(m);
}
