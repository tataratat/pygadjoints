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
class LinearElasticityProblem {
  // Typedefs
  typedef gsExprAssembler<>::geometryMap geometryMap;
  typedef gsExprAssembler<>::variable variable;
  typedef gsExprAssembler<>::space space;
  typedef gsExprAssembler<>::solution solution;

  // using SolverType = gsSparseSolver<>::CGDiagonal;
  using SolverType = gsSparseSolver<>::BiCGSTABILUT;

public:
  LinearElasticityProblem() : expr_assembler_pde(1, 1) {
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
  void SetMaterialConstants(const real_t &lambda, const real_t &mu,
                            const real_t &rho) {
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
    collection.addField(*solution_expression_ptr, "displacement");
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
    fd.getId(source_id, neumann_function);
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
    basis_function_ptr = std::make_shared<space>(
        expr_assembler_pde.getSpace(function_basis, dimensionality_));

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
    auto phys_jacobian = ijac(basis_function, geometric_mapping);
    auto bilin_lambda = lame_lambda_ * idiv(basis_function, geometric_mapping) *
                        idiv(basis_function, geometric_mapping).tr() *
                        meas(geometric_mapping);
    auto bilin_mu_1 = lame_mu_ *
                      (phys_jacobian.cwisetr() % phys_jacobian.tr()) *
                      meas(geometric_mapping);
    auto bilin_mu_2 = lame_mu_ * (phys_jacobian % phys_jacobian.tr()) *
                      meas(geometric_mapping);

    // Set the boundary_conditions term
    auto neumann_function_expression =
        expr_assembler_pde.getCoeff(neumann_function, geometric_mapping);
    auto lin_form = rho_ * basis_function * neumann_function_expression *
                    meas(geometric_mapping);

    auto bilin_combined = (bilin_lambda + bilin_mu_1 + bilin_mu_2);

    // Assemble
    expr_assembler_pde.assemble(bilin_combined);
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

  double ComputeVolume() {
    const Timer timer("ComputeVolume");

    // Compute volume of domain
    if (!expr_evaluator_ptr) {
      GISMO_ERROR("ExprEvaluator not initialized");
    }
    return expr_evaluator_ptr->integral(meas(*geometry_expression_ptr));
  }

  double ComputeObjectiveFunction() {
    const Timer timer("ComputeObjectiveFunction");
    if (!geometry_expression_ptr) {
      throw std::runtime_error("Error no geometry expression found.");
    }

    // Auxiliary
    solution &solution_expression = *solution_expression_ptr;

    // Integrate the compliance
    gsExprEvaluator<> expr_evaluator(expr_assembler_pde);
    real_t obj_value = expr_evaluator.integralBdrBc(
        boundary_conditions.get("Neumann"),
        (solution_expression.tr() * (solution_expression)) *
            nv(*geometry_expression_ptr).norm());

    return obj_value;
  }

  py::array_t<double> ComputeVolumeDerivativeToCTPS() {
    const Timer timer("ComputeVolumeDerivativeToCTPS");
    // Compute the derivative of the volume of the domain with respect to
    // the control points Auxiliary expressions
    const space &basis_function = *basis_function_ptr;
    auto jacobian = jac(*geometry_expression_ptr);      // validated
    auto inv_jacs = jacobian.ginv();                    // validated
    auto meas_expr = meas(*geometry_expression_ptr);    // validated
    auto djacdc = jac(basis_function);                  // validated
    auto aux_expr = (djacdc * inv_jacs).tr();           // validated
    auto meas_expr_dx = meas_expr * (aux_expr).trace(); // validated
    expr_assembler_pde.assemble(meas_expr_dx.tr());

    const auto volume_deriv =
        expr_assembler_pde.rhs().transpose() * (*ctps_sensitivities_matrix_ptr);

    py::array_t<double> derivative(volume_deriv.size());
    double *derivative_ptr = static_cast<double *>(derivative.request().ptr);
    for (int i{}; i < volume_deriv.size(); i++) {
      derivative_ptr[i] = volume_deriv(0, i);
    }
    return derivative;
  }

  void SolveAdjointProblem() {
    const Timer timer("SolveAdjointProblem");
    // Auxiliary references
    const geometryMap &geometric_mapping = *geometry_expression_ptr;
    const space &basis_function = *basis_function_ptr;
    const solution &solution_expression = *solution_expression_ptr;

    //////////////////////////////////////
    // Derivative of Objective Function //
    //////////////////////////////////////
    expr_assembler_pde.clearRhs();
    // Note that we assemble the negative part of the equation to avoid a
    // copy after solving
    expr_assembler_pde.assembleBdr(boundary_conditions.get("Neumann"),
                                   -2 * basis_function * solution_expression *
                                       nv(geometric_mapping).norm());
    const auto objective_function_derivative = expr_assembler_pde.rhs();

    /////////////////////////////////
    // Solving the adjoint problem //
    /////////////////////////////////
    const gsSparseMatrix<> matrix_in_initial_configuration(
        system_matrix->transpose().eval());
    auto rhs_vector = expr_assembler_pde.rhs();

    // Initialize linear solver
    gsSparseSolver<>::CGDiagonal solverAdjoint;
    solverAdjoint.compute(matrix_in_initial_configuration);
    // Minus is added in post
    lagrange_multipliers_ptr = std::make_shared<gsMatrix<>>(
        solverAdjoint.solve(expr_assembler_pde.rhs()));
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();
  }

  py::array_t<double> ComputeObjectiveFunctionDerivativeWrtCTPS() {
    const Timer timer("ComputeObjectiveFunctionDerivativeWrtCTPS");
    // Check if all required information is available
    if (!(geometry_expression_ptr && basis_function_ptr &&
          solution_expression_ptr && lagrange_multipliers_ptr)) {
      throw std::runtime_error(
          "Some of the required values are not yet initialized.");
    }

    if (!(ctps_sensitivities_matrix_ptr)) {
      throw std::runtime_error("CTPS Matrix has not been computed yet.");
    }

    // Auxiliary references
    const geometryMap &geometric_mapping = *geometry_expression_ptr;
    const space &basis_function = *basis_function_ptr;
    const solution &solution_expression = *solution_expression_ptr;

    ////////////////////////////////
    // Derivative of the LHS Form //
    ////////////////////////////////

    // Auxiliary expressions
    auto jacobian = jac(geometric_mapping);             // validated
    auto inv_jacs = jacobian.ginv();                    // validated
    auto meas_expr = meas(geometric_mapping);           // validated
    auto djacdc = jac(basis_function);                  // validated
    auto aux_expr = (djacdc * inv_jacs).tr();           // validated
    auto meas_expr_dx = meas_expr * (aux_expr).trace(); // validated

    // Start to assemble the bilinear form with the known solution field
    // 1. Bilinear form of lambda expression separated into 3 individual
    // sections
    auto BL_lambda_1 =
        idiv(solution_expression, geometric_mapping).val();     // validated
    auto BL_lambda_2 = idiv(basis_function, geometric_mapping); // validated
    auto BL_lambda =
        lame_lambda_ * BL_lambda_2 * BL_lambda_1 * meas_expr; // validated

    // trace(A * B) = A:B^T
    auto BL_lambda_1_dx = frobenius(
        aux_expr, ijac(solution_expression, geometric_mapping)); // validated
    auto BL_lambda_2_dx =
        (ijac(basis_function, geometric_mapping) % aux_expr); // validated

    auto BL_lambda_dx =
        lame_lambda_ * BL_lambda_2 * BL_lambda_1 * meas_expr_dx -
        lame_lambda_ * BL_lambda_2_dx * BL_lambda_1 * meas_expr -
        lame_lambda_ * BL_lambda_2 * BL_lambda_1_dx * meas_expr; // validated

    // 2. Bilinear form of mu (first part)
    // BL_mu1_2 seems to be in a weird order with [jac0, jac2] leading
    // to [2x(2nctps)]
    auto BL_mu1_1 = ijac(solution_expression, geometric_mapping); // validated
    auto BL_mu1_2 = ijac(basis_function, geometric_mapping);      // validated
    auto BL_mu1 = lame_mu_ * (BL_mu1_2 % BL_mu1_1) * meas_expr;   // validated

    auto BL_mu1_1_dx = -(ijac(solution_expression, geometric_mapping) *
                         aux_expr.cwisetr()); //          validated
    auto BL_mu1_2_dx =
        -(jac(basis_function) * inv_jacs * aux_expr.cwisetr()); // validated

    auto BL_mu1_dx0 =
        lame_mu_ * BL_mu1_2 % BL_mu1_1_dx * meas_expr; // validated
    auto BL_mu1_dx1 =
        lame_mu_ * frobenius(BL_mu1_2_dx, BL_mu1_1) * meas_expr; // validated
    auto BL_mu1_dx2 = lame_mu_ * frobenius(BL_mu1_2, BL_mu1_1).cwisetr() *
                      meas_expr_dx; // validated

    // 2. Bilinear form of mu (first part)
    auto BL_mu2_1 =
        ijac(solution_expression, geometric_mapping).cwisetr(); // validated
    auto &BL_mu2_2 = BL_mu1_2;                                  // validated
    auto BL_mu2 = lame_mu_ * (BL_mu2_2 % BL_mu2_1) * meas_expr; // validated

    auto inv_jac_T = inv_jacs.tr();
    auto BL_mu2_1_dx = -inv_jac_T * jac(basis_function).tr() * inv_jac_T *
                       jac(solution_expression).cwisetr(); // validated
    auto &BL_mu2_2_dx = BL_mu1_2_dx;                       // validated

    auto BL_mu2_dx0 =
        lame_mu_ * BL_mu2_2 % BL_mu2_1_dx * meas_expr; // validated
    auto BL_mu2_dx1 =
        lame_mu_ * frobenius(BL_mu2_2_dx, BL_mu2_1) * meas_expr; // validated
    auto BL_mu2_dx2 = lame_mu_ * frobenius(BL_mu2_2, BL_mu2_1).cwisetr() *
                      meas_expr_dx; // validated
    // Linear Form Part
    // auto LF_1 = -rho_ * basis_function * ff * meas_expr;
    // auto LF_1_dx = -rho_ * basis_function * ff * meas_expr_dx;

    // Assemble
    expr_assembler_pde.assemble(BL_lambda_dx + BL_mu1_dx0 + BL_mu1_dx2 +
                                    BL_mu2_dx0 + BL_mu2_dx2,
                                // + LF_1_dx,
                                BL_mu1_dx1, BL_mu2_dx1);

    ///////////////////////////
    // Compute sensitivities //
    ///////////////////////////
    const auto sensitivities =
        (lagrange_multipliers_ptr->transpose() * expr_assembler_pde.matrix()) *
        (*ctps_sensitivities_matrix_ptr);

    // Write eigen matrix into a py::array
    py::array_t<double> sensitivities_py(sensitivities.size());
    double *sensitivities_py_ptr =
        static_cast<double *>(sensitivities_py.request().ptr);
    for (int i{}; i < sensitivities.size(); i++) {
      sensitivities_py_ptr[i] = sensitivities(0, i);
    }
    // Clear for future evaluations
    expr_assembler_pde.clearMatrix(true);
    expr_assembler_pde.clearRhs();

    return sensitivities_py;
  }

  void
  GetParameterSensitivities(std::string filename // Filename for parametrization
  ) {
    const Timer timer("GetParameterSensitivities");
    gsFileData<> fd(filename);
    gsMultiPatch<> mp;
    fd.getId(0, mp);
    gsMatrix<index_t> patch_supports;
    fd.getId(10, patch_supports);

    const int design_dimension = patch_supports.col(1).maxCoeff() + 1;
    // h-refine each basis
    for (int r = 0; r < n_refinements; ++r) {
      mp.uniformRefine();
    }

    // Start the assignment
    if (!dof_mapper_ptr) {
      throw std::runtime_error("System has not been initialized");
    }

    // Start the assignment
    const size_t totalSz = dof_mapper_ptr->freeSize();
    ctps_sensitivities_matrix_ptr = std::make_shared<gsMatrix<>>();
    ctps_sensitivities_matrix_ptr->resize(totalSz, design_dimension);

    // Rough overestimate to avoid realloations
    for (int patch_support{}; patch_support < patch_supports.rows();
         patch_support++) {
      const int j_patch = patch_supports(patch_support, 0);
      const int i_design = patch_supports(patch_support, 1);
      const int k_index_offset = patch_supports(patch_support, 2);
      for (index_t k_dim = 0; k_dim != dimensionality_; k_dim++) {
        for (size_t l_dof = 0;
             l_dof != dof_mapper_ptr->patchSize(j_patch, k_dim); l_dof++) {
          if (dof_mapper_ptr->is_free(l_dof, j_patch, k_dim)) {
            const int global_id = dof_mapper_ptr->index(l_dof, j_patch, k_dim);
            ctps_sensitivities_matrix_ptr->operator()(global_id, i_design) =
                static_cast<double>(mp.patch(j_patch).coef(
                    l_dof, k_dim + k_index_offset * dimensionality_));
          }
        }
      }
    }
  }

  void UpdateGeometry(const std::string &fname, const bool &topology_changes) {
    const Timer timer("UpdateGeometry");
    if (topology_changes) {
      throw std::runtime_error("Not Implemented!");
    }

    // Import mesh and load relevant information
    gsMultiPatch<> mp_new;

    gsFileData<> fd(fname);
    fd.getId(0, mp_new);
    // Ignore all other information!
    if (mp_new.nPatches() != mp_pde.nPatches()) {
      throw std::runtime_error(
          "This does not work - I am fucked. Expected number of "
          "patches " +
          std::to_string(mp_pde.nPatches()) + ", but got " +
          std::to_string(mp_new.nPatches()));
    }
    // Manually update coefficients as to not overwrite any precomputed
    // values
    for (size_t patch_id{}; patch_id < mp_new.nPatches(); patch_id++) {
      if (mp_new.patch(patch_id).coefs().size() !=
          mp_pde.patch(patch_id).coefs().size()) {
        throw std::runtime_error(
            "This does not work - I am fucked. Expected number of "
            "coefficients " +
            std::to_string(mp_pde.patch(patch_id).coefs().size()) +
            ", but got " +
            std::to_string(mp_new.patch(patch_id).coefs().size()));
      }
      for (int i_coef = 0; i_coef != mp_pde.patch(patch_id).coefs().size();
           i_coef++) {
        mp_pde.patch(patch_id).coefs().at(i_coef) =
            mp_new.patch(patch_id).coefs().at(i_coef);
      }
    }
    // geometry_expression_ptr->copyCoefs(mp_new);
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

  /// Neumann function
  gsFunctionExpr<> neumann_function{};

  /// Function basis
  gsMultiBasis<> function_basis{};

  // Linear System Matrixn_refinements
  std::shared_ptr<const gsSparseMatrix<>> system_matrix = nullptr;

  // Linear System Matrixn_refinements
  std::shared_ptr<gsMatrix<>> ctps_sensitivities_matrix_ptr = nullptr;

  // Linear System RHS
  std::shared_ptr<gsMatrix<>> system_rhs = nullptr;

  // Solution of the adjoint problem
  std::shared_ptr<gsMatrix<>> lagrange_multipliers_ptr = nullptr;

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
