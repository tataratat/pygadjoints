"""
This is a simple diffusion problem

.. math:
  \\lambda \\Delta \theta + \bar{f} = 0

with boundary conditions

X ----- 3 ----- X
|               |
|               |
0               1
|               |
|               |
X ----- 1 ----- X

Neumann [3] with q = 10
Dirichlet [1] with T=0
Zero Neumann on [0], and [1]

and no source function
"""
import numpy as np
import splinepy as spp

import pygadjoints as pyg

# Define boundary conditions
lambda_ = 1.172e-5
density_ = 7850
thermal_capacity_ = 420
source_function_ = 1 / (density_ * thermal_capacity_)
neumann_flux_ = 10 / (density_ * thermal_capacity_)
dim = 2
print(f"Thermal Diffusivity : {lambda_}")

# Define function parameters
boundary_conditions_options = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {"type": "FunctionExpr", "id": "1", "dim": f"{dim}"},
        "text": "0.00001",
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
                    "dim": "2",
                    "index": "0",
                },
                "text": "0",
            },
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": "2",
                    "index": "1",
                },
                "text": f"{neumann_flux_} / ({thermal_capacity_} * {density_})",
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
        "text": "x*1.1 +1",
    },
]

# Set boundary conditions on all boundary elements of the multipatch (-1)
boundary_conditions_options[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Dirichlet",
            "function": str(0),
            "unknown": str(0),
            "name": f"BID{3}",
        },
    }
)
boundary_conditions_options[1]["children"].append(
    {
        "tag": "bc",
        "attributes": {
            "type": "Neumann",
            "function": str(1),
            "unknown": str(0),
            "name": f"BID{4}",
        },
    }
)

# Create a simple geometry
np.random.seed(1923879)
n_refine = 0
n_elevate = 1

# Create the geometry
microtile_patches = spp.helpme.create.box(3, 3).bspline
for _ in range(n_elevate):
    microtile_patches.elevate_degrees([0, 1])
microtile_patches.cps += 1 * np.random.random(microtile_patches.cps.shape)
microtile_patches.insert_knots(0, np.linspace(0, 1, n_refine + 2))
microtile_patches.insert_knots(1, np.linspace(0, 1, n_refine + 2))

# microtile_patches.show()
microtile = spp.Multipatch([microtile_patches])
microtile.determine_interfaces()
microtile.boundaries_from_continuity()
spp.io.gismo.export(
    "mini_example.xml",
    microtile,
    options=boundary_conditions_options,
    as_base64=True,
    labeled_boundaries=True,
)


diffusion_solver = pyg.DiffusionProblem()
diffusion_solver.init("mini_example.xml", 0, 0, False)
diffusion_solver.set_material_constants(lambda_)
# Original
diffusion_solver.assemble()
diffusion_solver.solve_linear_system()
diffusion_solver.solve_adjoint_system()
# diffusion_solver.print_debug()
sensitivities = diffusion_solver.objective_function_deris_wrt_ctps()
sensitivities_approx = np.zeros_like(sensitivities) * np.nan
original_obj_f = diffusion_solver.objective_function()

# Move ctps
step_size = 1e-4
n_dirichlet_ctps = microtile.patches[0].control_mesh_resolutions[0]
n_neumann_ctps = n_dirichlet_ctps
n_ctps = microtile.patches[0].cps.shape[0]
for i in range(0, 2):
    for j in range(n_dirichlet_ctps, n_ctps - n_neumann_ctps):
        microtile.patches[0].cps[j, i] += step_size

        # spp.show(microtile.patches[0])
        spp.io.gismo.export(
            "mini_example_dx.xml",
            microtile,
            options=boundary_conditions_options,
            as_base64=True,
            labeled_boundaries=True,
        )
        microtile.patches[0].cps[j, i] -= step_size

        # Disturbed
        diffusion_solver.update_geometry("mini_example_dx.xml", False)
        diffusion_solver.assemble()
        diffusion_solver.solve_linear_system()
        sensitivities_approx[
            i * n_dirichlet_ctps * 2 + j - n_dirichlet_ctps
        ] = (
            diffusion_solver.objective_function() - original_obj_f
        ) / step_size
sensitivities = sensitivities[~np.isnan(sensitivities_approx)]
sensitivities_approx = sensitivities_approx[~np.isnan(sensitivities_approx)]
print(f"Summary of the FD comparison with step-size {step_size} :")
print(f"Sensitivities : {sensitivities.tolist()}")
print(f"Approximation : {sensitivities_approx.tolist()}")
print(f"Diff. (abs)   : {abs(sensitivities - sensitivities_approx).tolist()}")
error = np.linalg.norm(sensitivities - sensitivities_approx)
print(f"Error (abs)   : {error}")
print(f"Error (rel)   : {error / np.linalg.norm(sensitivities)}")

diffusion_solver.export_paraview("solution", False, 80**dim, True)
diffusion_solver.export_xml("solution_2")
print("Export successful")
