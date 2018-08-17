import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start=-10, stop=10, num=1000)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

plt.plot(x, sigmoid(x))
plt.show()
