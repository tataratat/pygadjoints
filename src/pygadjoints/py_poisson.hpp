#include <gismo.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <chrono>

#ifdef PYGADJOINTS_USE_OPENMP
#include <omp.h>
#endif

#include "pygadjoints/custom_expression.hpp"

namespace pygadjoints {

using namespace gismo;

namespace py = pybind11;

class Timer {
#if 1
public:
  const std::string name;
  const std::chrono::time_point<std::chrono::high_resolution_clock>
      starting_time;
  Timer(const std::string &function_name)
      : name(function_name),
        starting_time{std::chrono::high_resolution_clock::now()} {
    std::cout << "[Timer] : " << std::setw(40) << name << "\tstarted..."
              << std::endl;
  }

  ~Timer() {
    std::cout << "[Timer] : " << std::setw(40) << name << "\tElapsed time : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - starting_time)
                     .count()
              << "ms" << std::endl;
  }
#endif
};

/// @brief
class PoissonProblem {
  // Typedefs
  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

  // using SolverType = gsSparseSolver<>::CGDiagonal;
  using SolverType = gsSparseSolver<>::BiCGSTABILUT;

public:
  PoissonProblem() : expr_assembler_pde(1, 1) {
#ifdef PYGADJOINTS_USE_OPENMP
    omp_set_num_threads(std::min(omp_get_max_threads(), n_omp_threads));
#endif
  };

  gsStopwatch timer;

  /**
   * @brief Export the results as Paraview vtu file
   *
   * @param fname Filename
   * @param plot_elements Plot patch borders
   * @param sample_rate Samplerate (samples per element)
   * @return void
   */
  void ExportParaview(const std::string &fname, const bool &plot_elements,
                      const int &sample_rate, const bool &export_b64) {
    // Generate Paraview File
    gsParaviewCollection collection("ParaviewOutput/" + fname,
                                    expr_evaluator_ptr.get());
    collection.options().setSwitch("plotElements", plot_elements);
    collection.options().setSwitch("base64", export_b64);
    collection.options().setInt("plotElements.resolution", sample_rate);
    collection.options().setInt("numPoints", sample_rate);
    collection.newTimeStep(&mp_pde);
    collection.addField(*solution_expression_ptr, "temperature");
    collection.saveTimeStep();
    collection.save();
  }

  /**
   * @brief Export the results in xml file
   *
   * @param fname Filename
   * @return double Elapsed time
   */
  double ExportXML(const std::string &fname) {
    // Export solution file as xml
    timer.restart();
    gsMultiPatch<> mpsol;
    gsMatrix<> full_solution;
    gsFileData<> output;
    output << solVector;
    solution_expression_ptr->extractFull(full_solution);
    output << full_solution;
    output.save(fname + ".xml");
    return timer.stop();
  }

#ifdef PYGADJOINTS_USE_OPENMP
  /**
   * @brief Set the Number Of Threads for OpenMP
   *
   * Somehow does not compile
   * @param n_threads
   */
  void SetNumberOfThreads(const int &n_threads) {
    n_omp_threads = n_threads;
    omp_set_num_threads(n_threads);
    // gsInfo << "Available threads: " << omp_get_max_threads() << "\n";
  }
#endif

  void ReadInputFromFile(const std::string &filename) {
    const Timer timer("ReadInputFromFile");
    // IDs in the text file (might change later)
    const int mp_id{0}, source_id{1}, bc_id{2}, ass_opt_id{3};

    // Import mesh and load relevant information
    gsFileData<> fd(filename);
    fd.getId(mp_id, mp_pde);
    fd.getId(source_id, source_function);
    fd.getId(bc_id, boundary_conditions);
    boundary_conditions.setGeoMap(mp_pde);
    gsOptionList Aopt;
    fd.getId(ass_opt_id, Aopt);

    // Set Options in expression assembler
    expr_assembler_pde.setOptions(Aopt);
  }

  void Init(const std::string &filename, const int numRefine) {
    const Timer timer("Init");
    // Read input parameters
    ReadInputFromFile(filename);

    // Set number of refinements
    n_refinements = numRefine;

    //! [Refinement]
    function_basis = gsMultiBasis<>(mp_pde, true);

    // h-refine each basis
    for (int r = 0; r < n_refinements; ++r) {
      function_basis.uniformRefine();
    }

    // Elements used for numerical integration
    expr_assembler_pde.setIntegrationElements(function_basis);

    // Set the dimension
    dimensionality_ = mp_pde.geoDim();

    // Set the discretization space
    basis_function_ptr =
        std::make_shared<space>(expr_assembler_pde.getSpace(function_basis, 1));

    // Solution vector and solution variable
    solution_expression_ptr = std::make_shared<solution>(
        expr_assembler_pde.getSolution(*basis_function_ptr, solVector));

    // Retrieve expression that represents the geometry mapping
    geometry_expression_ptr =
        std::make_shared<geometryMap>(expr_assembler_pde.getMap(mp_pde));

    basis_function_ptr->setup(boundary_conditions, dirichlet::l2Projection, 0);

    // Assign a Dof Mapper
    dof_mapper_ptr =
        std::make_shared<gsDofMapper>(basis_function_ptr->mapper());

    // Evaluator
    expr_evaluator_ptr = std::make_shared<gsExprEvaluator<>>(
        gsExprEvaluator<>(expr_assembler_pde));

    // Initialize the system
    expr_assembler_pde.initSystem();

    // Precalculate sensitity matrix
    GetParameterSensitivities(filename + ".fields.xml");
  }

  void Assemble() {
    const Timer timer("Assemble");
    if (!basis_function_ptr) {
      throw std::runtime_error("Error");
    }

    // Auxiliary variables for readability
    const geometryMap &geometric_mapping = *geometry_expression_ptr;
    const space &basis_function = *basis_function_ptr;

    // Compute the system matrix and right-hand side
    auto bilin = igrad(basis_function, geometric_mapping) *
                 igrad(basis_function, geometric_mapping).tr() *
                 meas(geometric_mapping);

    // Set the boundary_conditions term
    auto source_function_expression =
        expr_assembler_pde.getCoeff(source_function, geometric_mapping);
    auto lin_form =
        basis_function * source_function_expression * meas(geometric_mapping);

    // Assemble
    expr_assembler_pde.assemble(bilin);
    expr_assembler_pde.assemble(lin_form);

    // Compute the Neumann terms defined on physical space
    auto g_N = expr_assembler_pde.getBdrFunction(geometric_mapping);

    // Neumann conditions
    expr_assembler_pde.assembleBdr(boundary_conditions.get("Neumann"),
                                   basis_function * g_N *
                                       nv(geometric_mapping).norm());

    system_matrix =
        std::make_shared<const gsSparseMatrix<>>(expr_assembler_pde.matrix());
    system_rhs = std::make_shared<gsMatrix<>>(expr_assembler_pde.rhs());

    // Clear for future evaluations
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();
  }

  void SolveLinearSystem() {
    const Timer timer("SolveLinearSystem");

    ///////////////////
    // Linear Solver //
    ///////////////////
    if ((!system_matrix) || (!system_rhs)) {
      gsWarn << "System matrix and system rhs are required for solving!"
             << std::endl;
      return;
    }
    // Initialize linear solver
    SolverType solver;
    solver.compute(*system_matrix);
    solVector = solver.solve(*system_rhs);
  }

private:
  // -------------------------
  /// Expression assembler related to the forward problem
  gsExprAssembler<> expr_assembler_pde;

  /// Expression assembler related to the forward problem
  std::shared_ptr<gsExprEvaluator<>> expr_evaluator_ptr;

  /// Multipatch object of the forward problem
  gsMultiPatch<> mp_pde;

  /// Expression that describes the last calculated solution
  std::shared_ptr<solution> solution_expression_ptr = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<space> basis_function_ptr = nullptr;

  /// Expression that describes the last calculated solution
  std::shared_ptr<geometryMap> geometry_expression_ptr = nullptr;

  /// Global reference to solution vector
  gsMatrix<> solVector{};

  /// Boundary conditions pointer
  gsBoundaryConditions<> boundary_conditions;

  /// source function
  gsFunctionExpr<> source_function{};

  /// Function basis
  gsMultiBasis<> function_basis{};

  // Linear System Matrixn_refinements
  std::shared_ptr<const gsSparseMatrix<>> system_matrix = nullptr;

  // Linear System Matrixn_refinements
  std::shared_ptr<gsMatrix<>> ctps_sensitivities_matrix_ptr = nullptr;

  // Linear System RHS
  std::shared_ptr<gsMatrix<>> system_rhs = nullptr;

  // DOF-Mapper
  std::shared_ptr<gsDofMapper> dof_mapper_ptr = nullptr;

  // Number of refinements in the current iteration
  int n_refinements{};

  // Number of refinements in the current iteration
  int dimensionality_{};

#ifdef PYGADJOINTS_USE_OPENMP
  int n_omp_threads{1};
#endif
};

} // namespace pygadjoints
