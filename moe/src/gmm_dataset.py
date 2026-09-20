import torch 
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def generate_dataset(M, D, E, batch_size):

    means = np.random.rand(E) * 100
    variances = np.random.rand(E) * 3


    X, Y = [], []
    for i in range(M):
        cluster_choice = np.random.randint(0, E)
        x = np.random.normal(loc=means[cluster_choice], scale=variances[cluster_choice], size=D)
        y = cluster_choice
        X.append(x)
        Y.append(y)

    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)

    assert X.shape == (M, D)
    assert Y.shape == (M, )

    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.savefig('gmm_dataset.png') 
    plt.show()

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    return dataloader

