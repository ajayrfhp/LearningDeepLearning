import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
x = np.empty(0)
y = np.empty(0)
for i in range(1, 10):
    xx = np.linspace(i * 10, (i + 1) * 10, 100)
    yy = np.sin(xx)
    x = np.concatenate((x, xx))
    y = np.concatenate((y, yy))
    ax.clear()
    ax.plot(x, y)
    plt.pause(0.5)

plt.show()
