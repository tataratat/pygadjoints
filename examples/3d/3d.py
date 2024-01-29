if __name__ == "__main__":
    from numpy import array as Array, linspace as Linspace
    from pygadjoints import LinearElasticityProblem
    from scipy.optimize import minimize as Minimize
    from splinepy import BSpline
    from splinepy.helpme import create, permute
    from splinepy.microstructure import Microstructure, tiles
    from splinepy.io import irit, gismo

    # Configure parameters of optimization problem
    shape, tiling, tile, number_of_optimization_parameters_per_direction, optimization_parameter_initial, \
            scaling_of_objective_function, number_of_refinements = create.box(5, 3, 2).bspline, [1, 1, 2],
                    tiles.Cross3D(), 2, 0.2, 100.0, 0
    shape.insert_knot(0, [0.5])
    permute.parametric_axes(shape, permutation_list=[0, 2, 1])

    knot_vector, number_of_optimization_parameters = [0.0, *Linspace(0.0, 1.0,
            number_of_optimization_parameters_per_direction), 1.0], number_of_optimization_parameters_per_direction**3
    optimization_parameters_initial, gismo_file_name = \
            Array([[optimization_parameter_initial]] * number_of_optimization_parameters), \
            "box" + str(tiling[0]) + "Times" + str(tiling[1]) + "Times" + str(tiling[2]) + ".xml"
    parameter_spline = BSpline(degrees=[1] * 3, knot_vectors=[knot_vector] * 3,
                               control_points=optimization_parameters_initial)

    # Prepare geometry for simulation
    optimization_parameters_stored, interfaces_stored = None, None
    def PrepareSimulationForCurrentDesign(optimization_parameters):
        from numpy import allclose as AllClose, put_along_axis as PutAlongAxis, zeros as Zeros

        global optimization_parameters_stored, interfaces_stored

        if optimization_parameters_stored is not None:
                if AllClose(optimization_parameters, optimization_parameters_stored): return
        # else: create design for current optimization parameters
        optimization_parameters_stored = optimization_parameters.copy()
        parameter_spline.control_points[:] = optimization_parameters.reshape(-1, 1)
        microstructure = Microstructure(deformation_function=shape, tiling=tiling, microtile=tile,
                                        parametrization_function=lambda parametric_coordinates :
                                                parameter_spline.evaluate(parametric_coordinates))
        def ParameterSensitivityFunction(parametric_coordinates):
            number_of_parametric_coordinates = parametric_coordinates.shape[0]
            basis_function_matrix = Zeros((number_of_parametric_coordinates, number_of_optimization_parameters))
            basis_functions, support = parameter_spline.basis_and_support(parametric_coordinates)
            PutAlongAxis(basis_function_matrix, support, basis_functions, axis=1)
            return basis_function_matrix.reshape(number_of_parametric_coordinates, 1, -1)

        microstructure.parameter_sensitivity_function = ParameterSensitivityFunction
#        multi_patch = microstructure.create(closing_face="z")

        from splinepy.multipatch import Multipatch as MultiPatch

        shape.elevate_degrees([0, 1, 2])
        beziers_multi_patch = shape.extract.beziers()
        beziers_multi_patch[1].control_points[-1][0] += 1.0e-5
        multi_patch = MultiPatch(beziers_multi_patch)
        beziers = shape.extract.beziers()
        for bezier in beziers:
            bezier.control_points[:] = 0.0

        beziers[1].control_points[-1][0] = 1.0
        multi_patch.add_fields([beziers], 3)

        if interfaces_stored is None:
            for boundary_id in (2, ): multi_patch.boundary_from_function(lambda points : shape.extract.boundaries(
                    boundary_ids=[boundary_id])[0].proximities(queries=points,
                            initial_guess_sample_resolutions=[10, 10], tolerance=1e-9,
                                    return_verbose=True)[3].flatten() < 1e-8)
            interfaces_stored = multi_patch.interfaces
        else:
            multi_patch.interfaces = interfaces_stored

        data = [
            {
                "tag": "Function",
                "attributes": {"type": "FunctionExpr", "id": "1", "dim": "3", "c": "3"},
                "children": [
                    {
                        "tag": "c",
                        "attributes": {"index": "0"},
                        "text": "0.0"
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "1"},
                        "text": "0.0"
                    },
                    {
                        "tag": "c",
                        "attributes": {"index": "2"},
                        "text": "1.09662e6 * (0.35 + z)"  # 10k rpm => 10k pi / 30 radian => square root of 1.09663e6
                    }
                ]
            },
            {
                "tag": "boundaryConditions",
                "attributes": {"id": "2", "multipatch": "0"},
                "children": [
                     {
                        "tag": "Function",
                        "attributes": {"type": "FunctionExpr", "index": "0", "dim": "3", "c": "3"},
                        "children": [
                            {
                                "tag": "c",
                                "attributes": {"index": "0"},
                                "text": "0.0"
                            },
                            {
                                "tag": "c",
                                "attributes": {"index": "1"},
                                "text": "0.0"
                            },
                            {
                                "tag": "c",
                                "attributes": {"index": "2"},
                                "text": "0.0"
                            }
                        ]
                     },
                     {
                        "tag": "Function",
                        "attributes": {"type": "FunctionExpr", "index": "1", "dim": "3", "c": "3"},
                        "children": [
                            {
                                "tag": "c",
                                "attributes": {"index": "0"},
                                "text": "1.6e1"
                            },
                            {
                                "tag": "c",
                                "attributes": {"index": "1"},
                                "text": "9.95e2"
                            },
                            {
                                "tag": "c",
                                "attributes": {"index": "2"},
                                "text": "0.0"
                            }
                        ]
                    },
                    {
                        "tag": "bc",
                        "attributes": {"unknown": "0", "type": "Dirichlet", "function": "0", "name": "BID2"}
                    }#,
        #            {
        #                "tag": "bc",
        #                "attributes": {"unknown": "0", "type": "Neumann", "function": "1", "name": "BID4"}
        #            }
                ]
            }#,
#            {
#                "tag": "OptionList",
#                "attributes": {"id": "3"},
#                "text": "\n    ",
#                "children": [
#                    {
#                        "tag": "int",
#                        "attributes": {
#                            "label": "DirichletStrategy",
#                            "desc": "Method for enforcement of Dirichlet BCs [11..14]",
#                            "value": "11"
#                        }
#                    },
#                    {
#                        "tag": "int",
#                        "attributes": {
#                            "label": "DirichletValues",
#                            "desc": "Method for computation of Dirichlet DoF values [100..103]",
#                            "value": "100"
#                        }
#                    },
#                    {
#                        "tag": "int",
#                        "attributes": {
#                            "label": "InterfaceStrategy",
#                            "desc": "Method of treatment of patch interfaces [0..3]",
#                            "value": "1"
#                        }
#                    },
#                    {
#                        "tag": "real",
#                        "attributes": {
#                            "label": "bdA",
#                            "desc": "Estimated nonzeros per column of the matrix: bdA*deg + bdB",
#                            "value": "78.0"
#                        }
#                    },
#                    {
#                        "tag": "int",
#                        "attributes": {
#                            "label": "bdB",
#                            "desc": "Estimated nonzeros per column of the matrix: bdA*deg + bdB",
#                            "value": "1"
#                        }
#                    },
#                    {
#                        "tag": "real",
#                        "attributes": {
#                            "label": "bdO",
#                            "desc": "Overhead of sparse mem. allocation: (1+bdO)(bdA*deg + bdB) [0..1]",
#                            "value": "0.33334"
#                        }
#                    },
#                    {
#                        "tag": "real",
#                        "attributes": {
#                            "label": "quA",
#                            "desc": "Number of quadrature points: quA*deg + quB; For patchRule: Regularity of the target space",
#                            "value": "1.0"
#                        }
#                    },
#                    {
#                        "tag": "int",
#                        "attributes": {
#                            "label": "quB",
#                            "desc": "Number of quadrature points: quA*deg + quB; For patchRule: Degree of the target space",
#                            "value": "1"
#                        }
#                    },
#                    {
#                        "tag": "int",
#                        "attributes": {
#                            "label": "quRule",
#                            "desc": "Quadrature rule used (1) Gauss-Legendre; (2) Gauss-Lobatto; (3) Patch-Rule",
#                            "value": "1"
#                        }
#                    },
#                    {
#                        "tag": "bool",
#                        "attributes": {
#                            "label": "overInt",
#                            "desc": "Apply over-integration on boundary elements?",
#                            "value": "false"
#                        }
#                    },
#                    {
#                        "tag": "bool",
#                        "attributes": {
#                            "label": "flipSide",
#                            "desc": "Flip side of interface where integration is performed?",
#                            "value": "false"
#                        }
#                    },
#                    {
#                        "tag": "bool",
#                        "attributes": {
#                            "label": "movingInterface",
#                            "desc": "Is interface not stationary?",
#                            "value": "false"
#                        }
#                    }
#                ]
#            }
        ]
        gismo.export(gismo_file_name, multi_patch, indent=True, labeled_boundaries=True, options=data,
                     export_fields=True, as_base64=True)

    PrepareSimulationForCurrentDesign(optimization_parameters_initial)

    # Prepare forward solves during optimization
    linear_elasticity_solver = LinearElasticityProblem()
    linear_elasticity_solver.set_material_constants(1.2e11, 8.0e10, 8.22e3)
    linear_elasticity_solver.set_number_of_threads(4)
    linear_elasticity_solver.init(gismo_file_name, number_of_refinements)

    # Prepare optimization
    objective_function_value_stored = None
    def UpdateCurrentDesign(optimization_parameters):
        global objective_function_value_stored

        PrepareSimulationForCurrentDesign(optimization_parameters)
        linear_elasticity_solver.update_geometry(gismo_file_name, topology_changes=False)
        optimization_parameters_stored, objective_function_value_stored = optimization_parameters.copy(), None

    def ObjectiveFunction(optimization_parameters):
        global objective_function_value_stored

        PrepareSimulationForCurrentDesign(optimization_parameters)
        if objective_function_value_stored is not None: return objective_function_value_stored
        # else: compute objective function for new design
        linear_elasticity_solver.assemble()
        linear_elasticity_solver.solve_linear_system()
        objective_function_value_stored = linear_elasticity_solver.objective_function() * scaling_of_objective_function
        print(objective_function_value_stored, '\n')
        return objective_function_value_stored

    def ObjectiveFunctionJacobian(optimization_parameters):
        ObjectiveFunction(optimization_parameters)
        linear_elasticity_solver.solve_adjoint_system()
        print(linear_elasticity_solver.objective_function_deris_wrt_ctps() * scaling_of_objective_function, '\n')
        return linear_elasticity_solver.objective_function_deris_wrt_ctps() * scaling_of_objective_function

    def Volume(optimization_parameters):
        PrepareSimulationForCurrentDesign(optimization_parameters)
        print(linear_elasticity_solver.volume(), '\n')
        return (4.0e-6 - linear_elasticity_solver.volume())

    def VolumeChange(optimization_parameters):
        PrepareSimulationForCurrentDesign(optimization_parameters)
        print(-linear_elasticity_solver.volume_deris_wrt_ctps(), '\n')
        return -linear_elasticity_solver.volume_deris_wrt_ctps()

    minimum = Minimize(fun=ObjectiveFunction, x0=optimization_parameters_initial.ravel(), method="SLSQP",
                       jac=ObjectiveFunctionJacobian, bounds=[(0.05, 0.35)] * number_of_optimization_parameters,
                       constraints={"type" : "ineq", "fun" : Volume, "jac" : VolumeChange}, options={"disp" : True})
    linear_elasticity_solver.export_paraview("solution.pvd", plot_elements=False, sample_rate=100, binary=True)
    print("Optimal parameters\n{}\nafter minimization with the following outcome\n{}".format(minimum.x, minimum))
    exit()
