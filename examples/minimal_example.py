from timeit import default_timer as timer

import pygadjoints

# Current test-file
filename = "/home/zwar/Git/pygadjoints/examples/lattice_structure_4x4.xml"
# filename = "/home/zwar/Git/pygadjoints/examples/lattice_structure_24x12.xml"


# Initialize the problem (needs to be done only once)
linear_solver = pygadjoints.LinearElasticityProblem()
linear_solver.set_number_of_threads(8)

# Treat input files
linear_solver.read_from_input_file(filename)
linear_solver.init(1)
# m_data = linear_solver.read_control_point_sensitivities(filename + ".fields.xml")
# matrix = csr_matrix(m_data[0], shape=m_data[1])

# First assembly
start = timer()
linear_solver.assemble()
print(f"Assembly time first assembly {timer() - start}")
start = timer()
linear_solver.solve_linear_system()
print(f"Solving time first assembly {timer() - start}")

# Second assembly
start = timer()
linear_solver.assemble()
print(f"Assembly time second assembly {timer() - start}")
start = timer()
linear_solver.solve_linear_system()
print(f"Solving time second assembly {timer() - start}")

exit()
linear_solver.export_xml("test_xml.xml")
linear_solver.export_paraview("solution", False, 100, False)
print(linear_solver.volume())
print(linear_solver.volume_deris_wrt_ctps().shape)
print(linear_solver.objective_function())
