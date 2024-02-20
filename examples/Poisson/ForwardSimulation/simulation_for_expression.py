import numpy as np
import splinepy as spp

import pygadjoints as pyg


"""
This is a simple diffusion problem

.. math:
  \lambda \Delta \theta + \bar{f} = 0

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

# Define boundary conditions
lambda_ = 1.172e-5
density_ = 7850
thermal_capacity_ = 420
dim = 2
print(f"Thermal Diffusivity : {lambda_}")

# Define function parameters
boundary_conditions_options = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {"type": "FunctionExpr", "id": "1", "dim": f"{dim}"},
        "text": "0",
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
                    "dim": f"2",
                    "index": "0",
                },
                "text": "0",
            },
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": f"2",
                    "index": "1",
                },
                "text": f"10 / ({thermal_capacity_} * {density_})",
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
microtile.patches[0].cps[0, 1] += 1e-4

# spp.show(microtile.patches[0])
spp.io.gismo.export(
    "mini_example_dx.xml",
    microtile,
    options=boundary_conditions_options,
    as_base64=True,
    labeled_boundaries=True,
)


diffusion_solver = pyg.DiffusionProblem()
diffusion_solver.init("mini_example.xml", 0,  0, False)
diffusion_solver.set_material_constants(lambda_)
# Original
diffusion_solver.assemble()
diffusion_solver.solve_linear_system()
diffusion_solver.solve_adjoint_system()
diffusion_solver.objective_function_deris_wrt_ctps()

# Move ctps

# Disturbed
# diffusion_solver.update_geometry("mini_example_dx.xml", False)
# diffusion_solver.objective_function_deris_wrt_ctps()

print(diffusion_solver.objective_function())
diffusion_solver.export_paraview("solution", False, 80**dim, True)
diffusion_solver.export_xml("solution_2")
print("Export successfull")
