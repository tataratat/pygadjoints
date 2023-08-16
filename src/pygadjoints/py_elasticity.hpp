#include <gismo.h>

#ifdef PYGADJOINTS_USE_OPENMP
#include <omp.h>
#endif

#include "pygadjoints/custom_expression.hpp"

namespace pygadjoints {

using namespace gismo;

/// @brief
class LinearElasticityProblem {
  // Typedefs
  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

 public:
  LinearElasticityProblem() {
    expr_assembler_pde = std::make_shared<gsExprAssembler<>>(1, 1);
    mp_pde = std::make_shared<gsMultiPatch<>>();
    boundary_conditions = std::make_shared<gsBoundaryConditions<>>();
#ifdef PYGADJOINTS_USE_OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
#endif
  };

  gsStopwatch timer;

  /**
   * @brief Set the Material Constants
   *
   * @param lambda first lame constant
   * @param mu second lame constant
   * @param rho density
   */
  void SetMaterialConstants(const real_t& lambda, const real_t& mu,
                            const real_t& rho) {
    lame_lambda_ = lambda;
    lame_mu_ = mu;
    rho_ = rho;
  }

  /**
   * @brief Export the results as Paraview vtu file
   *
   * @param fname Filename
   * @param plot_elements Plot patch borders
   * @param sample_rate Samplerate (samples per element)
   * @return double Elapsed time
   */
  double ExportParaview(const std::string& fname, const bool& plot_elements,
                        const int& sample_rate) {
    // Generate Paraview File
    gsExprEvaluator<> expression_evaluator(*expr_assembler_pde);
    timer.restart();
    gsParaviewCollection collection("ParaviewOutput/" + fname,
                                    &expression_evaluator);
    collection.options().setSwitch("plotElements", true);
    collection.options().setInt("numPoints", plot_elements);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.newTimeStep(mp_pde.get());
    collection.addField(*solution_expression, "displacement");
    collection.saveTimeStep();
    collection.save();
    return timer.stop();
  }

  /**
   * @brief Export the results in xml file
   *
   * @param fname Filename
   * @return double Elapsed time
   */
  double ExportXML(const std::string& fname) {
    // Export solution file as xml
    timer.restart();
    gsMultiPatch<> mpsol;
    gsMatrix<> full_solution;
    gsFileData<> output;
    output << solVector;
    solution_expression->extractFull(full_solution);
    output << full_solution;
    output.save(fname + ".xml");
    return timer.stop();
  }

  void print(const int i) {
    for (int j{}; j < i; j++) {
      gsWarn << "TEST IF GISMO IS THERE SUCCEEDED, I AM YOUR\n";
    }
  }

#ifdef PYGADJOINTS_USE_OPENMP
  /**
   * @brief Set the Number Of Threads for OpenMP
   *
   * Somehow does not compile
   * @param n_threads
   */
  void SetNumberOfThreads(const int& n_threads) {
    n_omp_threads = n_threads;
    omp_set_num_threads(n_threads);
    // gsInfo << "Available threads: " << omp_get_max_threads() << "\n";
  }
#endif

  void ReadInputFromFile(const std::string& filename) {
    // IDs in the text file (might change later)
    const int mp_id{0}, source_id{1}, bc_id{2}, ass_opt_id{3};

    // Import mesh and load relevant information
    gsFileData<> fd(filename);
    fd.getId(mp_id, *mp_pde);
    fd.getId(source_id, neumann_function);
    fd.getId(bc_id, *boundary_conditions);
    boundary_conditions->setGeoMap(*mp_pde);
    gsOptionList Aopt;
    fd.getId(ass_opt_id, Aopt);

    // Set Options in expression assembler
    std::cout << "Test 1";
    expr_assembler_pde->setOptions(Aopt);
  }

  void Init(const int numRefine) {
    //! [Refinement]
    std::cout << "Test 1";
    if (!mp_pde) {
      std::cout << ("Multipatch is not initialized!");
      return;
    }
    function_basis = std::make_shared<gsMultiBasis<>>(*mp_pde, true);

    // h-refine each basis
    for (int r = 0; r < numRefine; ++r) {
      function_basis->uniformRefine();
    }

    // Elements used for numerical integration
    expr_assembler_pde->setIntegrationElements(*function_basis);

    // Set the geometry map
    geometric_mapping_ptr =
        std::make_shared<geometryMap>(expr_assembler_pde->getMap(*mp_pde));

    // Set the discretization space
    basis_function_ptr = std::make_shared<space>(std::move(
        expr_assembler_pde->getSpace(*function_basis, mp_pde->geoDim())));

    // Solution vector and solution variable
    solution_expression = std::make_shared<solution>(
        expr_assembler_pde->getSolution(*basis_function_ptr, solVector));
    basis_function_ptr->setup(*boundary_conditions, dirichlet::l2Projection, 0);

    // Initialize the system
    expr_assembler_pde->initSystem();
  }

  void Assemble() {
    if ((!expr_assembler_pde) || (!basis_function_ptr) ||
        (!geometric_mapping_ptr)) {
      std::cerr << "ERROR";
      return;
    }

    // Auxiliary variables for readability
    const geometryMap& geometric_mapping = *geometric_mapping_ptr;
    const space& basis_function = *basis_function_ptr;

    // Compute the system matrix and right-hand side
    auto phys_jacobian = ijac(basis_function, geometric_mapping);
    auto bilin_lambda = lame_lambda_ * idiv(basis_function, geometric_mapping) *
                        idiv(basis_function, geometric_mapping).tr() *
                        meas(geometric_mapping);
    auto bilin_mu_1 = lame_mu_ *
                      (phys_jacobian.cwisetr() % phys_jacobian.tr()) *
                      meas(geometric_mapping);
    auto bilin_mu_2 = lame_mu_ * (phys_jacobian % phys_jacobian.tr()) *
                      meas(geometric_mapping);

    // Set the bc term
    auto neumann_function_expression =
        expr_assembler_pde->getCoeff(neumann_function, *geometric_mapping_ptr);
    auto lin_form = rho_ * basis_function * neumann_function_expression *
                    meas(geometric_mapping);

    auto bilin_combined = (bilin_lambda + bilin_mu_1 + bilin_mu_2);

    // Assemble
    expr_assembler_pde->assemble(bilin_combined);
    expr_assembler_pde->assemble(lin_form);

    // Compute the Neumann terms defined on physical space
    auto g_N = expr_assembler_pde->getBdrFunction(geometric_mapping);

    // Neumann conditions
    expr_assembler_pde->assembleBdr(
        boundary_conditions->get("Neumann"),
        basis_function * g_N * nv(geometric_mapping).norm());
  }

  void SolveLinearSystem() {
    ///////////////////
    // Linear Solver //
    ///////////////////
    timer.restart();
    const auto& matrix_in_initial_configuration = expr_assembler_pde->matrix();

    // Initialize linear solver
    gsSparseSolver<>::CGDiagonal solver;
    solver.compute(matrix_in_initial_configuration);
    solVector = solver.solve(expr_assembler_pde->rhs());
  }

 private:
  // -------------------------
  /// First Lame constant
  real_t lame_lambda_{2000000};
  /// Second Lame constant
  real_t lame_mu_{500000};
  /// Density
  real_t rho_{1000};

  // -------------------------
  /// Expression assembler related to the forward problem
  std::shared_ptr<gsExprAssembler<>> expr_assembler_pde = nullptr;

  /// Multipatch object of the forward problem
  std::shared_ptr<gsMultiPatch<>> mp_pde = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<solution> solution_expression = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<space> basis_function_ptr = nullptr;

  /// Global reference to solution vector
  gsMatrix<> solVector{};

  /// Boundary conditions pointer
  std::shared_ptr<gsBoundaryConditions<>> boundary_conditions;

  /// Neumann function
  gsFunctionExpr<> neumann_function;

  /// Geometric Map
  std::shared_ptr<geometryMap> geometric_mapping_ptr = nullptr;
  std::shared_ptr<gsMultiBasis<>> function_basis = nullptr;

#ifdef PYGADJOINTS_USE_OPENMP
  int n_omp_threads{1};
#endif
};

}  // namespace pygadjoints
