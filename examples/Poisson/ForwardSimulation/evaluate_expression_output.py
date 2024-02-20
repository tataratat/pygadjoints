import numpy as np

dx = 1e-4
first = np.genfromtxt("examples/Poisson/ForwardSimulation/output_first.csv")
second = np.genfromtxt("examples/Poisson/ForwardSimulation/output_second.csv")
third = np.genfromtxt("examples/Poisson/ForwardSimulation/output_third.csv")
data = np.genfromtxt("examples/Poisson/ForwardSimulation/output.csv")
print(first.shape)

print(second.shape)
print(third.shape)
print(data.shape)

print(np.allclose(first+second+third, data))
# deriv = 1e4 * (first-second)

# print(deriv)
# print(data[:, 18:20].T)

# print((deriv - data[:, 18:20].T))
