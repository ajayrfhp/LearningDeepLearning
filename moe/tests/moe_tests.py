import torch
import sys, os

sys.path.append(os.path.dirname(__file__) + "/../src")

from shazeer_moe import NoisyTopKGating, ShazeerMOE
import gmm_dataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
 

def fit_batch(moe, train_dataloader, test_dataloader, criterion, a = 0.01, num_epochs=2):
    optim = torch.optim.Adam(moe.parameters(), lr=1e-3)

    losses = []
    for _ in range(num_epochs):
        for (X, target_tensor) in train_dataloader:
            optim.zero_grad()
            pred, aux_loss = moe.forward(X)
            loss = criterion(pred.reshape((-1, X.shape[-1])), target_tensor.reshape((-1, X.shape[-1])))
            total_loss = loss + aux_loss * a

            total_loss.backward()
            optim.step()

            losses.append(loss.item())

    correct = 0 
    total = 0
    for (X_test, Y_test) in test_dataloader:
        preds = moe.forward(X_test)
        assert preds.shape == (M, D_out)
        preds = preds.argmax(dim=-1)
        batch_correct = (preds == Y_test).sum()
        correct += batch_correct
        total += X_test.shape[0]

    acc = correct / total 
    print(f"acc {acc}")


    plt.plot(range(len(losses)), losses)
    plt.savefig('loss_plot.png') 
    plt.close()

def fit(moe, X, target_tensor, a = 0.01, num_epochs=2):
    optim = torch.optim.Adam(moe.parameters(), lr=1e-3)

    losses = []
    for _ in range(num_epochs):
        optim.zero_grad()
        pred, aux_loss = moe.forward(X)
        loss = torch.nn.MSELoss()(pred.reshape((-1, X.shape[-1])), target_tensor.reshape((-1, X.shape[-1])))
        total_loss = loss + aux_loss * a

        total_loss.backward()
        optim.step()

        losses.append(loss.item())

    plt.plot(range(len(losses)), losses)
    plt.savefig('loss_plot.png') 
    plt.close()
    


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
    Y = (X.clone().detach())  * 3 
    # Y = Y.unsqueeze(-1).expand(-1, -1, D)
    assert Y.shape == (B, S, D)
    Y = Y.reshape((M, D))

    assert Y.shape == (M, D)


    moe = ShazeerMOE(D=D, N=N, K=K)
    fit(moe, X, Y, num_epochs=10000, a = 0)

    new_inpt = torch.tensor([[[1]]]).to(X)

    pred, _ = moe.forward(new_inpt)

    expected_gt = torch.tensor([[3]]).to(pred)

    assert torch.allclose(pred[0], expected_gt, atol=0.5), f"{pred[0].item()} is not 3"

def test_router_collapse():
    X = torch.randn((B, S, D))
    Y = (X.clone().detach())  * 3 
    # Y = Y.unsqueeze(-1).expand(-1, -1, D)
    assert Y.shape == (B, S, D)
    Y = Y.reshape((M, D))

    assert Y.shape == (M, D)



    for aux_loss_penalty in [0, 1, 100, 10000]:
        moe_router_collapse = ShazeerMOE(D=D, N=N, K=K)

        moe_router_collapse.noisy_gating.W_G.data[:,0] = 10 # assign large weight to expert 0

        fit(moe_router_collapse, X, Y, num_epochs=10000, a=aux_loss_penalty) 

        router_sums = moe_router_collapse.noisy_gating.W_G.sum(dim=0)

        print(f"{aux_loss_penalty} router sum {router_sums}")


def test_gmm_fit():
    moe = ShazeerMOE(D_in=D_in, N=N, K=K)
    train_dataloader, test_dataloader = gmm_dataset.generate_dataset(M, D_in, N, batch_size=B)
    criterion = torch.nn.CrossEntropyLoss()
    fit_batch(moe, train_dataloader, test_dataloader, criterion)


if __name__ == "__main__":
    B = 1000
    S = 5
    D_in = 2
    D_out = 4

    N = 4
    K = 3
    M = B * S

    # test_non_zero_gradient()
    # test_synthetic_overfitting()
    # test_router_collapse()
    test_gmm_fit()
