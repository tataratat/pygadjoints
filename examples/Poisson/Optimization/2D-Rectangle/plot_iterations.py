import matplotlib.pyplot as plt
import numpy as np
array = np.genfromtxt("log_file_iterations.csv", delimiter=",")
plt.subplot(2, 1, 1)
plt.plot(array[:, 1].flat)
plt.title('Evolution of the Objective function')
plt.ylabel('Objective Funtion')
step_size = np.linalg.norm(np.diff(array[:, 2:], axis=0), axis=1)

plt.subplot(2, 1, 2)
plt.semilogy(np.hstack((step_size, [0])))
plt.xlabel('Iteration')
plt.ylabel('Step size')

plt.show()
