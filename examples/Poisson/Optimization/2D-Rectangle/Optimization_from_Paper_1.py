import numpy as np
import scipy
import splinepy as sp

import pygadjoints

###
# SIMULATION PARAMETERS
###
ULTRA_VERBOSE = True
INVALID_ID = 1999
N_THREAD = 4

###
# MATERIAL PARAMETERS
###
ACTIVATE_SOURCE_FUNCTION = False
thermal_conductivity = 1
density_ = 1
thermal_capacity_ = 1
lambda_ = thermal_conductivity / (density_ * thermal_capacity_)  # 1.172e-5
source_function_ = 0 / (density_ * thermal_capacity_)
dirichlet_value = 0
dim = 2

print(f"Thermal Diffusivity : {lambda_}")

# Define function parameters
GISMO_OPTIONS = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "id": f"{1}",
            "dim": f"{dim}",
        },
        "text": "0.",
    },
    {
        # Boundary Conditions
        "tag": "boundaryConditions",
        "attributes": {"multipatch": "0", "id": "2"},
        "children": [
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "0",
                },
                "text": f"{dirichlet_value}",
            },
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"{dim}",
                    "index": "1",
                },
                "text": "27 / 256 * (4 - x) * x * x + 1",
            },
        ],
    },
    {
        # Solution Expression
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "dim": f"{dim}",
            "id": "3",
        },
        "text": "0",
    },
    {
        # Target Solution Expression
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "dim": f"{dim}",
            "id": "11",
        },
        "text": "10 + 5 * x / 4",
    },
]

# Set boundary conditions on all boundary elements of the multipatch (-1)
GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Dirichlet",
            "function": str(0),
            "unknown": str(0),
            "name": f"BID{4}",
        },
    }
)

GISMO_OPTIONS[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Neumann",
            "function": str(1),
            "unknown": str(0),
            "name": f"BID{5}",
        },
    }
)

if not ACTIVATE_SOURCE_FUNCTION:
    GISMO_OPTIONS.pop(0)


# for i in range(1, 8):
#     GISMO_OPTIONS[1]["children"].append(
#         {
#             "tag": "bc",
#             "attributes": {
#                 "type": "Dirichlet",
#                 "function": str(i + 2),
#                 "unknown": str(0),
#                 "name": f"BID{i}",
#             },
#         }
#     )


class Optimizer:
    def __init__(
        self,
        macro_spline,
        microtile,
        para_spline,
        identifier_function_neumann,
        tiling=[24, 12],
        scaling_factor_objective_function=100,
        n_refinements=1,
        write_logfiles=False,
        max_volume=1.5,
        macro_ctps=None,
        parameter_default_value=0.1,
        volume_scaling=1,
        micro_structure_keys=None,
    ):
        self.parameter_default_value = parameter_default_value
        self.n_refinements = n_refinements
        self.microtile = microtile
        self.interfaces = None
        self.macro_spline = macro_spline.bspline
        self.macro_spline_original = self.macro_spline.copy()
        self.para_spline = para_spline.bspline
        self.identifier_function_neumann = identifier_function_neumann
        self.tiling = tiling
        self.scaling_factor_objective_function = (
            scaling_factor_objective_function
        )
        self.diffusion_solver = pygadjoints.DiffusionProblem()
        self.diffusion_solver.set_number_of_threads(N_THREAD)
        self.diffusion_solver.set_material_constants(lambda_)
        self.last_parameters = None
        self.iteration = 0
        self.write_logfiles = write_logfiles
        self.max_volume = max_volume
        self.macro_ctps = macro_ctps
        self._volume_scaling = volume_scaling
        self._ms_keys = (
            dict() if micro_structure_keys is None else micro_structure_keys
        )

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

        multipatch = generator.create(
            contact_length=0.5,
            macro_sensitivities=len(self.macro_ctps) > 0,
            **self._ms_keys,
        )

        # Reuse existing interfaces
        if self.interfaces is None:
            multipatch.determine_interfaces()
            print("Number of control-points in geometry")
            print(np.sum([s.cps.shape[0] for s in multipatch.patches]))
            for i in range(self.macro_spline.dim * 2):
                multipatch.boundary_from_function(
                    identifier_function(generator.deformation_function, i)
                )
            if self.identifier_function_neumann is not None:
                multipatch.boundary_from_function(
                    self.identifier_function_neumann, mask=[5]
                )

            self.interfaces = multipatch.interfaces
        else:
            multipatch.interfaces = self.interfaces
        sp.io.gismo.export(
            self.get_filename(),
            multipatch=multipatch,
            options=GISMO_OPTIONS,
            export_fields=True,
            as_base64=True,
            field_mask=(
                np.arange(0, self.para_spline.cps.shape[0]).tolist()
                + (
                    np.array(self.macro_ctps) + self.para_spline.cps.shape[0]
                ).tolist()
            ),
        )

    def ensure_parameters(self, parameters):
        # Check if anything changed since last call
        if self.last_parameters is not None and np.allclose(
            self.last_parameters, parameters
        ):
            return
        self.iteration += 1
        # Something differs (or first iteration)
        self.para_spline.cps[:] = parameters[
            : self.para_spline.cps.shape[0]
        ].reshape(-1, 1)
        self.macro_spline.cps.ravel()[self.macro_ctps] = (
            parameters[self.para_spline.cps.shape[0]:]
            + self.macro_spline_original.cps.ravel()[self.macro_ctps]
        )
        self.prepare_microstructure()
        if self.last_parameters is None:
            # First iteration
            self.diffusion_solver.init(
                self.get_filename(), self.n_refinements, 0, True
            )
            self.diffusion_solver.read_control_point_sensitivities(
                self.get_filename() + ".fields.xml"
            )
            self.control_point_sensitivities = (
                self.diffusion_solver.get_control_point_sensitivities()
            )
            self.last_parameters = parameters.copy()
        else:
            self.diffusion_solver.update_geometry(
                self.get_filename(), topology_changes=False
            )
            self.diffusion_solver.read_control_point_sensitivities(
                self.get_filename() + ".fields.xml"
            )
            self.control_point_sensitivities = (
                self.diffusion_solver.get_control_point_sensitivities()
            )
            self.last_parameters = parameters.copy()

        # Notify iteration evaluator
        self.current_objective_function_value = None
        self.ctps_sensitivity = None

    def evaluate_iteration(self, parameters):
        self.ensure_parameters(parameters)
        if self.current_objective_function_value is not None:
            return self.current_objective_function_value

        # There is no current solution all checks have been performed
        self.diffusion_solver.assemble()
        self.diffusion_solver.solve_linear_system()
        self.current_objective_function_value = (
            self.diffusion_solver.objective_function()
            * self.scaling_factor_objective_function
        )

        # Write into logfile
        with open("log_file_iterations.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + [self.current_objective_function_value]
                        + parameters.tolist()
                    )
                )
                + "\n"
            )

        return self.current_objective_function_value

    def evaluate_jacobian(self, parameters):
        # Make sure that current file is valid
        _ = self.evaluate_iteration(parameters)

        # Determine Lagrange multipliers
        self.diffusion_solver.solve_adjoint_system()
        ctps_sensitivities = (
            self.diffusion_solver.objective_function_deris_wrt_ctps()
        )
        parameter_sensitivities = (
            ctps_sensitivities @ self.control_point_sensitivities
        ) * self.scaling_factor_objective_function

        # Write into logfile
        with open("log_file_sensitivities.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + parameter_sensitivities.tolist()
                        + parameters.tolist()
                    )
                )
                + "\n"
            )
        return parameter_sensitivities

    def volume(self, parameters):
        self.ensure_parameters(parameters)
        volume = self.diffusion_solver.volume() * self._volume_scaling

        # Write into logfile
        with open("log_file_volume.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration] + [volume] + parameters.tolist()
                    )
                )
                + "\n"
            )

        return self.max_volume - volume

    def volume_deriv(self, parameters):
        self.ensure_parameters(parameters)
        sensi = (
            -self.diffusion_solver.volume_deris_wrt_ctps()
            * self._volume_scaling
        )

        # Write into logfile
        with open("log_file_volume_sensitivities.csv", "a") as file1:
            file1.write(
                ", ".join(
                    str(a)
                    for a in (
                        [self.iteration]
                        + (-sensi).tolist()
                        + parameters.tolist()
                    )
                )
                + "\n"
            )
        return sensi

    def constraint(self):
        return {"type": "ineq", "fun": self.volume, "jac": self.volume_deriv}

    def finalize(self):
        self.diffusion_solver.export_paraview("solution", False, 100, True)

    def optimize(self):
        # Initialize the optimization
        n_design_vars_para = self.para_spline.cps.size
        n_design_vars_macro = len(self.macro_ctps)
        initial_guess = np.empty(n_design_vars_macro + n_design_vars_para)
        initial_guess[:n_design_vars_para] = (
            np.ones(n_design_vars_para) * self.parameter_default_value
        )
        initial_guess[n_design_vars_para:] = 0

        optim = scipy.optimize.minimize(
            self.evaluate_iteration,
            initial_guess.ravel(),
            method="SLSQP",
            jac=self.evaluate_jacobian,
            bounds=(
                [(0.0111, 0.249) for _ in range(n_design_vars_para)]
                + [(-0.5, 0.5) for _ in range(n_design_vars_macro)]
            ),
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
    tiles_with_load = 2
    tiling = [10, 1]
    parameter_spline_degrees = [1, 1]
    parameter_spline_cps_dimensions = [4, 2]
    parameter_default_value = 0.15

    scaling_factor_objective_function = 0.01

    sp.settings.NTHREADS = 1

    write_logfiles = True

    # Create parameters spline
    parameter_spline = sp.BSpline(
        degrees=parameter_spline_degrees,
        knot_vectors=[
            (
                [0] * parameter_spline_degrees[i]
                + np.linspace(
                    0,
                    1,
                    parameter_spline_cps_dimensions[i]
                    - parameter_spline_degrees[i]
                    + 1,
                ).tolist()
                + [1] * parameter_spline_degrees[i]
            )
            for i in range(len(parameter_spline_degrees))
        ],
        control_points=np.ones((np.prod(parameter_spline_cps_dimensions), 1))
        * parameter_default_value,
    )

    # Function for neumann boundary
    identifier_function_neumann = None
    macro_spline = sp.Bezier(
        degrees=[1, 1],
        control_points=[
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 2.0],
            [4.0, 2.0],
        ],
    ).bspline
    closure_thickness = 0.05
    macro_spline.insert_knots(1, np.linspace(
        closure_thickness, 1 - closure_thickness, 6))

    volume_scaling = 1 / macro_spline.integrate.volume()
    print(f"Max Volume is:{macro_spline.integrate.volume()}")

    optimizer = Optimizer(
        microtile=sp.microstructure.tiles.Cross2D(),
        macro_spline=macro_spline,
        para_spline=parameter_spline,
        identifier_function_neumann=None,
        tiling=tiling,
        scaling_factor_objective_function=scaling_factor_objective_function,
        n_refinements=1,
        write_logfiles=write_logfiles,
        max_volume=0.5,
        macro_ctps=[],
        parameter_default_value=parameter_default_value,
        volume_scaling=volume_scaling,
        micro_structure_keys={"center_expansion": 1.2, "closing_face": "y"},
    )

    # Try some parameters
    optimizer.optimize()
    optimizer.finalize()

    exit()


if "__main__" == __name__:
    main()
