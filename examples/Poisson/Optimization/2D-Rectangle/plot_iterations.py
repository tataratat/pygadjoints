import matplotlib.pyplot as plt
import numpy as np
iterations = np.genfromtxt("log_file_iterations.csv", delimiter=",")
sensitivities = np.genfromtxt("log_file_sensitivities.csv", delimiter=",")
n_params = int((sensitivities.shape[1]-2)/2)

plt.subplot(3, 1, 1)
plt.plot(iterations[:, 1].flat, '-o')
plt.title('Evolution of the Objective function')
plt.ylabel('Objective Funtion')
step_size = np.linalg.norm(np.diff(iterations[:, 2:], axis=0), axis=1)


plt.subplot(3, 1, 2)
plt.semilogy(np.hstack((step_size, [0])))
plt.xlabel('Iteration')
plt.ylabel('Step size')

plt.subplot(3, 1, 3)
plt.plot(sensitivities[:, 0], np.linalg.norm(
    sensitivities[:, 2:(2+n_params)], axis=1), '-x')
plt.xlabel('Norm of sensitivities')
plt.ylabel('Step size')

plt.show()
