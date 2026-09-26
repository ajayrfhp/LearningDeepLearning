import torch

class SimpleDNN(torch.nn.Module):
    def __init__(self, D, H, N):
        super(SimpleDNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D, H),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            # torch.nn.Dropout(),
            torch.nn.Linear(H, H),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            # torch.nn.Dropout(),
            torch.nn.Linear(H, H),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            # torch.nn.Dropout(),
            torch.nn.Linear(H, N)
        )

    def forward(self, x):
        return self.net.forward(x)