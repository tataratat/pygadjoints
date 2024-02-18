import numpy as np
import splinepy as spp

import pygadjoints as pyg


"""
This example solves a thermal diffusion problem in the form

.. math:
  \lambda \Delta \theta + \bar{f} = 0

We construct the manufactured solution with two variables in the form of 

.. math:
  \theta = \alpha \text{sin} (2x) + \beta x y^2

which can be obtained, iff :math:`\bar{f}` is given as

.. math:
  \bar{f} = \lambda (4 \text{sin} (2x) + 2\beta x)

"""

# Define boundary conditions
lambda_ = 1.172e-5
print(f"Thermal Diffusivity : {lambda_}")

# Define function parameters
alpha_ = 2
beta_ = 3.1

boundary_conditions_options = [
    {
        # F - function (source)
        "tag": "Function",
        "attributes": {"type": "FunctionExpr", "id": "1", "dim": "3"},
        "text":
        f"4 * sin(2 * x) * {alpha_} * {lambda_} -"
        f" 2 * {beta_} * x * {lambda_}",
    },
    {
        "tag": "boundaryConditions",
        "attributes": {"multipatch": "0", "id": "2"},
        "children": [
            {
                "tag": "Function",
                "attributes": {
                    "type": "FunctionExpr",
                    "dim": "3",
                    "index": "0",
                },
                "text":
                f"sin(2 * x) * {alpha_} "
                f" + {beta_} * x * y * y",
            },
        ],
    },
    {
        "tag": "Function",
        "attributes": {
            "type": "FunctionExpr",
            "dim": "3",
            "id": "3",
        },
        "text":
        f"sin(2 * x) * {alpha_} "
        f"+ {beta_} * x * y * y",
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
            "name": f"BID{1}",
        },
    }
)

# Create a simple geometry
np.random.seed(9234865)
n_refine = 2
n_elevate = 1

microtile_patches = spp.helpme.create.box(3, 4, 3.5).bspline
for _ in range(n_elevate):
    microtile_patches.elevate_degrees([0, 1, 2])
microtile_patches.cps += 1 * np.random.random(microtile_patches.cps.shape)
microtile_patches.insert_knots(0, np.linspace(0, 1, n_refine ** 2))
microtile_patches.insert_knots(1, np.linspace(0, 1, n_refine ** 2))
microtile_patches.insert_knots(2, np.linspace(0, 1, n_refine ** 2))
# microtile_patches.show()
microtile = spp.Multipatch([microtile_patches])
microtile.determine_interfaces()
# microtile.boundaries_from_continuity()
spp.io.gismo.export(
    "mini_example.xml",
    microtile,
    options=boundary_conditions_options,
    as_base64=False,
    labeled_boundaries=True,
)
print("Test 1")

diffusion_solver = pyg.DiffusionProblem()
diffusion_solver.init("mini_example.xml", 0, True)
diffusion_solver.set_material_constants(lambda_)
diffusion_solver.assemble()
diffusion_solver.solve_linear_system()
diffusion_solver.export_paraview("solution", False, 80**3, True)
diffusion_solver.export_xml("solution_2")
print("Export successfull")
