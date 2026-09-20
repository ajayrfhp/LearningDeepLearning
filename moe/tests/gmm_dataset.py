import torch 
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

B = 100
D = 2
E = 4



means, variances = [], []
for _ in range(E):
    mean = np.random.rand() * 100 
    variance = np.random.rand()* 2
    means.append(mean)
    variances.append(variance)



X, Y = [], []
for i in range(B):
    cluster_choice = np.random.randint(0, E)
    x = np.random.normal(loc=means[cluster_choice], scale=variances[cluster_choice], size=D)
    y = cluster_choice
    X.append(x)
    Y.append(y)

X, Y = np.array(X), np.array(Y)

assert X.shape == (B, D)
assert Y.shape == (B, )

plt.scatter(X[:,0], X[:,1], c=Y)
plt.savefig('gmm_dataset.png') 
plt.show()
