import pygadjoints
from timeit import default_timer as timer


linear_solver = pygadjoints.LinearElasticityProblem()
linear_solver.read_from_input_file("lattice_structure_4x4.xml")
linear_solver.init(1)
linear_solver.set_number_of_threads(8)
start = timer()
# First assembly
linear_solver.assemble()
end = timer()
print(end - start) 
start = timer()
# Second assembly
linear_solver.assemble()
end = timer()
print(end - start) 
linear_solver.solve_linear_system()
linear_solver.export_
