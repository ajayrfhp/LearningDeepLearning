import torch 
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def generate_dataset(M, D, N, batch_size, test_size=0.2):

    means = np.random.rand(N) * 100
    variances = np.random.rand(N) * 3


    X, Y = [], []
    for i in range(M):
        cluster_choice = np.random.randint(0, N)
        x = np.random.normal(loc=means[cluster_choice], scale=variances[cluster_choice], size=D)
        y = cluster_choice
        X.append(x)
        Y.append(y)

    X, Y = np.array(X), np.array(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    train_length = int(test_size * X.shape[0])
    X_train, X_test, Y_train, Y_test = X[:train_length], X[train_length:], Y[:train_length], Y[train_length:]

    X_train, Y_train = torch.from_numpy(X_train), torch.from_numpy(Y_train)
    X_test, Y_test = torch.from_numpy(X_test), torch.from_numpy(Y_test)
    

    assert X.shape == (M, D), f"shape of X is instead {X.shape}"
    assert Y.shape == (M, )

    assert X_train.shape[0] == Y_train.shape[0]
    assert X_test.shape[0] == Y_test.shape[0]

    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.savefig('gmm_dataset.png') 
    plt.show()

    train_dataset = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size)

    return train_dataloader, test_dataloader

