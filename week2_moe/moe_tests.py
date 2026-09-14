import torch
from moe import NoisyTopKGating, ShazeerMOE

def fit(moe, X, target_tensor, a = 0.01, num_epochs=2):
    optim = torch.optim.Adam(moe.parameters(), lr=1e-3)

    for _ in range(num_epochs):
        optim.zero_grad()
        pred, aux_loss = moe.forward(X)
        loss = torch.nn.MSELoss()(pred, target_tensor)
        total_loss = loss + aux_loss * a

        total_loss.backward()
        optim.step()
    


def test_non_zero_gradient():
    moe = ShazeerMOE(D=D, N=N, K=K)

    X = torch.randn((B, S, D))
    Y = torch.randn((M, D))

    fit(moe, X, Y)

    # grad should be non zero for W_G, W_N and W_EK

    parameters = {item[0] : item[1].grad for item in moe.named_parameters() if item[1].grad is not None}

    w_g = parameters['noisy_gating.W_G']
    w_n = parameters['noisy_gating.W_N']
    w_e = parameters['experts']

    assert w_g.sum().abs() > 0 and w_n.sum().abs() > 0 and w_e.sum().abs() > 0

def test_synthetic_overfitting():
    X = torch.randn((B, S, D))
    Y = torch.tensor(X.clone().detach().sum(dim=-1)) 
    Y = Y.unsqueeze(-1).expand(-1, -1, D)
    assert Y.shape == (B, S, D)
    Y = Y.reshape((M, D))

    assert Y.shape == (M, D)


    moe = ShazeerMOE(D=D, N=N, K=K)
    fit(moe, X, Y, num_epochs=100, a = 0)

    print(X[0][0])
    print(Y[0])

    new_inpt = torch.tensor([[[1, 0]]]).to(X)
    pred, _ = moe.forward(new_inpt)

    print(new_inpt)
    print(pred)

    expected_gt = torch.tensor([[1]]).to(pred)

    assert torch.allclose(pred[0], expected_gt)

    


if __name__ == "__main__":
    B = 100
    S = 5
    D = 2

    N = 4
    K = 3
    M = B * S

    test_non_zero_gradient()
    test_synthetic_overfitting()
