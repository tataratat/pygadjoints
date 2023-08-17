#include "pygadjoints/py_elasticity.hpp"
using namespace pygadjoints;
int main() {
  LinearElasticityProblem linear_elasticity_problem{};
  linear_elasticity_problem.ReadInputFromFile("lattice_structure_4x4.xml");
  linear_elasticity_problem.Init(0);
  std::cout << "SUCCESS :party:" << std::flush;

  return 0;
}
