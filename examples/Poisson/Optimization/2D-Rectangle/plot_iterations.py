import matplotlib.pyplot as plt
import numpy as np

iterations = np.genfromtxt("log_file_iterations.csv", delimiter=",")
try:
    volumes = np.genfromtxt("log_file_volume.csv", delimiter=",")
    has_volumes = True
    max_plots = 4
except:
    has_volumes = False
    max_plots = 3

sensitivities = np.genfromtxt("log_file_sensitivities.csv", delimiter=",")
n_params = int((sensitivities.shape[1] - 2) / 2)
steps_with_sensitivities = sensitivities[:, 0].astype(np.int64) - 1

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
step_size = np.linalg.norm(np.diff(iterations[:, 2:], axis=0), axis=1)


plt.subplot(max_plots, 1, 2)
plt.semilogy(iterations[:, 0].flat, np.hstack((step_size, [0])), "-x")
plt.xlabel("Iteration")
plt.ylabel("Step size")

plt.subplot(max_plots, 1, 3)
plt.plot(
    sensitivities[:, 0],
    np.linalg.norm(sensitivities[:, 2 : (2 + n_params)], axis=1),
    "-x",
)
plt.ylabel("Norm of sensitivities")
plt.xlabel("Iteration")

if has_volumes:
    plt.subplot(max_plots, 1, 4)
    plt.plot(volumes[:, 0], volumes[:, 1], "-x")
    plt.ylabel("Norm of sensitivities")
    plt.xlabel("Iteration")

plt.show()
