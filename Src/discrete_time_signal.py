import numpy as np
import matplotlib.pyplot as plt

#time samples (index)


n = np.arange(-10, 11)

#discrete time signal

x = np.cos((np.pi/4)*n)

#plot the signal

plt.stem(n, x)
plt.title("Discrete Time Signal")
plt.xlabel("time index")
plt.ylabel("x[n]")
plt.grid(True)
#plt.savefig(r"C:Users\Umons\Documents\Github\Digital Signal Processing\Results\Discrete_time_signal.png")
plt.show()
