import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Method 4: Using add_subplot for more control of subplot locations.
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)  # 2 rows, 1 col, subplot 1
ax1.plot(x, y1, label="Plot 1")
ax1.set_title("Top Plot")
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)  # 2 rows, 1 col, subplot 2
ax2.plot(x, y2, label="Plot 2")
ax2.set_title("Bottom Plot")
ax2.legend()

print("ax1:", ax1)
print("ax2:", ax2)

plt.tight_layout()  # Improves subplot spacing
plt.show()
