import splinepy as sp
from options import gismo_options
import pygadjoints

geometry = sp.BSpline(
    degrees=[2, 2],
    control_points=[
        [0.0, 0.0],
        [1.0, 0.5],
        [2.0, 0.2],
        [0.5, 1.5],
        [1.0, 1.5],
        [1.5, 1.5],
        [0.0, 3.0],
        [1.0, 2.5],
        [2.0, 3.0],
    ],
    knot_vectors=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
)



def main():
    # Set the number of available threads (will be passed to splinepy and
    # pygdjoints)
    sp.io.gismo.export(fname="poisson",
        multipatch=geometry,
        options=gismo_options,
        export_fields=True,
        as_base64=True,
    )
    n_threads = 12

    linear_solver = pygadjoints.PoissonProblem()
    linear_solver.set_number_of_threads(n_threads)
    linear_solver.assemble()
    linear_solver.solve_linear_system()
    linear_solver.export_paraview("solution", False, 100, True)

    exit()


if "__main__" == __name__:
    main()
