import matplotlib.pyplot as plt
import numpy as np


def piecewise_function(x):
    return np.piecewise(x, [x <= 2, x > 2], [lambda x: x + 2, lambda x: -4 * (x - 2) + 4])

x_values = np.linspace(1, 3, 100)
y_values = piecewise_function(x_values)

plt.figure(figsize=(8, 6))

plt.plot(x_values, y_values)


plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0])
plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sample graph!')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

plt.legend()
plt.show()