import matplotlib.pyplot as plt
import numpy as np

iterations = np.genfromtxt("log_file_iterations.csv", delimiter=",")
try:
    volumes = np.genfromtxt("log_file_volume.csv", delimiter=",")
    has_volumes = True
    max_plots = 4
except Exception:
    has_volumes = False
    max_plots = 3

sensitivities = np.genfromtxt("log_file_sensitivities.csv", delimiter=",")
steps_with_sensitivities = sensitivities[:, 0].astype(np.int64) - 1
step_size_sensitivities = step_size = np.linalg.norm(
    np.diff(iterations[steps_with_sensitivities, 2:], axis=0), axis=1
)

#
n_params = int((sensitivities.shape[1] - 2) / 2)
max_iter = int(iterations[-1, 0])
major_grid_x = 5

plt.subplot(max_plots, 1, 1)
plt.semilogy(iterations[:, 0].flat, iterations[:, 1].flat, "-x")
plt.semilogy(
    iterations[steps_with_sensitivities, 0].flat,
    iterations[steps_with_sensitivities, 1].flat,
    "o",
)
plt.title("Evolution of the Objective function")
plt.ylabel("Objective Function")
plt.ylim((np.min(iterations[:, 1]) * 0.8, iterations[0, 1] * 1.2))
plt.xlim((0, max_iter))
plt.xticks(list(range(0, max_iter, major_grid_x)))
step_size = np.linalg.norm(np.diff(iterations[:, 2:], axis=0), axis=1)


plt.subplot(max_plots, 1, 2)
plt.semilogy(iterations[:-1, 0].ravel() + 0.5, step_size, [0], "-x")
plt.semilogy(
    sensitivities[:-1, 0].ravel() + 0.5, step_size_sensitivities.flat, "o"
)
plt.xticks(list(range(0, max_iter, major_grid_x)))
plt.xlim((0, max_iter))
plt.xlabel("Iteration")
plt.ylabel("Step size")

plt.subplot(max_plots, 1, 3)
plt.plot(
    sensitivities[:, 0],
    np.linalg.norm(sensitivities[:, 2 : (2 + n_params)], axis=1),
    "-x",
)
plt.xticks(list(range(0, max_iter, major_grid_x)))
plt.ylabel("Norm of sensitivities")
plt.xlim((0, max_iter))
plt.xlabel("Iteration")

if has_volumes:
    plt.subplot(max_plots, 1, 4)
    plt.plot(volumes[:, 0], volumes[:, 1], "-x")
    plt.ylabel("Norm of sensitivities")
    plt.xlabel("Iteration")
    plt.xticks(list(range(0, max_iter, major_grid_x)))
    plt.xlim((0, max_iter))

plt.show()
