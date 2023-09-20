from timeit import default_timer as timer

import numpy as np
import scipy
import splinepy as sp
from options import gismo_options

import pygadjoints


class Optimizer:
    def __init__(
        self,
        macro_spline,
        microtile,
        para_spline,
        identifier_function_neumann,
        n_threads=12,
        tiling=[24, 12],
        scaling_factor_objective_function=100,
        n_refinements=1,
    ):
        self.n_refinements = n_refinements
        self.microtile = microtile
        self.interfaces = None
        self.macro_spline = macro_spline.bspline
        self.para_spline = para_spline.bspline
        self.identifier_function_neumann = identifier_function_neumann
        self.tiling = tiling
        self.scaling_factor_objective_function = (
            scaling_factor_objective_function
        )
        self.linear_solver = pygadjoints.LinearElasticityProblem()
        self.linear_solver.set_number_of_threads(n_threads)
        self.last_parameters = None

    def prepare_microstructure(self):
        def parametrization_function(x):
            """
            Parametrization Function (determines thickness)
            """
            return self.para_spline.evaluate(x)

        def parameter_sensitivity_function(x):
            basis_function_matrix = np.zeros(
                (x.shape[0], self.para_spline.control_points.shape[0])
            )
            basis_functions, support = self.para_spline.basis_and_support(x)
            np.put_along_axis(
                basis_function_matrix, support, basis_functions, axis=1
            )
            return basis_function_matrix.reshape(x.shape[0], 1, -1)

        # Initialize microstructure generator and assign values
        generator = sp.microstructure.Microstructure()
        generator.deformation_function = self.macro_spline
        generator.tiling = self.tiling
        generator.microtile = self.microtile
        generator.parametrization_function = parametrization_function
        generator.parameter_sensitivity_function = (
            parameter_sensitivity_function
        )

        # Creator for identifier functions
        def identifier_function(deformation_function, face_id):
            boundary_spline = deformation_function.extract.boundaries()[
                face_id
            ]

            def identifier_function(x):
                distance_2_boundary = boundary_spline.proximities(
                    queries=x,
                    initial_guess_sample_resolutions=[4],
                    tolerance=1e-9,
                    return_verbose=True,
                )[3]
                return distance_2_boundary.flatten() < 1e-8

            return identifier_function

        multipatch = generator.create(contact_length=0.5)

        # Reuse existing interfaces
        if self.interfaces is None:
            multipatch.determine_interfaces()
            for i in range(self.macro_spline.dim * 2):
                multipatch.boundary_from_function(
                    identifier_function(generator.deformation_function, i)
                )
            multipatch.boundary_from_function(
                self.identifier_function_neumann, mask=[5]
            )
            self.interfaces = multipatch.interfaces
        else:
            multipatch.interfaces = self.interfaces
        sp.io.gismo.export(
            self.get_filename(),
            multipatch=multipatch,
            options=gismo_options,
            export_fields=True,
            as_base64=True,
        )

    def ensure_parameters(self, parameters):
        # Check if anything changed since last call
        if self.last_parameters is not None and np.allclose(
            self.last_parameters, parameters
        ):
            return

        # Something differs (or first iteration)
        self.para_spline.cps[:] = parameters.reshape(-1, 1)
        self.prepare_microstructure()
        if self.last_parameters is None:
            # First iteration
            self.linear_solver.init(self.get_filename(), self.n_refinements)
            self.last_parameters = parameters.copy()
        else:
            self.linear_solver.update_geometry(
                self.get_filename(), topology_changes=False
            )
            self.last_parameters = parameters.copy()
            # self.linear_solver.read_from_input_file(self.get_filename())
            self.last_parameters = parameters.copy()

        # Notify iteration evaluator
        self.current_objective_function_value = None
        self.ctps_sensitivity = None

    def evaluate_iteration(self, parameters):
        self.ensure_parameters(parameters)
        if self.current_objective_function_value is not None:
            return self.current_objective_function_value

        # There is no current solution all checks have been performed
        timer()
        self.linear_solver.assemble()
        # print(f"Assembly : {timer() - start }s")

        timer()
        self.linear_solver.solve_linear_system()
        # print(f"Solving  : {timer() - start }s")
        self.current_objective_function_value = (
            self.linear_solver.objective_function()
            * self.scaling_factor_objective_function
        )
        # self.finalize()
        print(
            f"Params : {parameters} Value : {self.current_objective_function_value}"
        )
        return self.current_objective_function_value

    def evaluate_jacobian(self, parameters):
        # Make sure that current file is valid
        _ = self.evaluate_iteration(parameters)

        # Determine Lagrange multipliers
        self.linear_solver.solve_adjoint_system()
        ctps_sensitivities = (
            self.linear_solver.objective_function_deris_wrt_ctps()
            * self.scaling_factor_objective_function
        )
        return ctps_sensitivities

    def volume(self, parameters):
        self.ensure_parameters(parameters)
        volume = 1.5 - self.linear_solver.volume()
        # print(f"volume is {volume}")
        return volume

    def volume_deriv(self, parameters):
        self.ensure_parameters(parameters)
        sensi = -self.linear_solver.volume_deris_wrt_ctps()
        return sensi

    def constraint(self):
        return {"type": "ineq", "fun": self.volume, "jac": self.volume_deriv}

    def finalize(self):
        self.linear_solver.export_paraview("solution", False, 100, True)

    def optimize(self):
        n_design_vars = self.para_spline.cps.size
        initial_guess = np.ones((n_design_vars, 1)) * 0.145

        optim = scipy.optimize.minimize(
            self.evaluate_iteration,
            initial_guess.ravel(),
            method="SLSQP",
            jac=self.evaluate_jacobian,
            bounds=[(0.0111, 0.249) for _ in range(n_design_vars)],
            constraints=self.constraint(),
            options={"disp": True},
        )
        # Finalize
        self.finalize()
        print("Best Parameters : ")
        print(optim.x)
        print(optim)

    def get_filename(self):
        return (
            "lattice_structure_"
            + str(self.tiling[0])
            + "x"
            + str(self.tiling[1])
            + ".xml"
        )


def main():
    # Set the number of available threads (will be passed to splinepy and
    # pygdjoints)

    # Geometry definition
    length = 2
    height = 1
    tiling = [12, 4]

    def identifier_function_neumann(x):
        return x[:, 0] >= (tiling[0] - 1) / tiling[0] * length - 1e-12

    macro_spline = sp.helpme.create.box(length, height)
    parameter_spline = sp.BSpline(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 0.25, 0.5, 0.75, 1, 1], [0, 0, 0.5, 1, 1]],
        control_points=np.ones((5 * 3, 1)) * 0.1,
    )
    optimizer = Optimizer(
        microtile=sp.microstructure.tiles.DoubleLatticeTile(),
        macro_spline=macro_spline,
        para_spline=parameter_spline,
        identifier_function_neumann=identifier_function_neumann,
        tiling=tiling,
        scaling_factor_objective_function=100,
        n_refinements=1,
    )
    # Try some parameters
    optimizer.optimize()
    optimizer.finalize()

    exit()


if "__main__" == __name__:
    main()
