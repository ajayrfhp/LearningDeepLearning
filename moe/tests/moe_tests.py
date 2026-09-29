import torch
import sys, os

sys.path.append(os.path.dirname(__file__) + "/../src")

from shazeer_moe import NoisyTopKGating, ShazeerMOE
from simple_dnn import SimpleDNN
import gmm_dataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torchinfo import summary
import wandb 

def fit_batch(model, train_dataloader, test_dataloader, criterion, D_out, num_epochs=2, a=1):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    aux_losses = []
    if torch.cuda.is_available():
        model.to("cuda")
    
    for i in range(num_epochs):
        model.train()
        for (X, target_tensor) in train_dataloader:
            if torch.cuda.is_available():
                X = X.to("cuda")
                model.to("cuda")
                target_tensor = target_tensor.to("cuda")

            optim.zero_grad()
            aux_loss = 0
            if isinstance(model, ShazeerMOE):
                pred, aux_loss = model.forward(X)
            else:
                pred = model.forward(X)
            loss = criterion(pred, target_tensor) + a * aux_loss
            loss.backward()
            optim.step()

            if isinstance(model, SimpleDNN):
                weight0 = model.net[0].weight
                weightlast = model.net[-1].weight
                # print(f"weight 0 norm {weight0.norm()} gradweight0 norm {weight0.grad.norm()}")
                # print(f"weight last norm {weightlast.norm()} gradweightlast norm {weightlast.grad.norm()}")
            # print(f"loss {loss.item()}")

            if isinstance(model, ShazeerMOE):
                expert_weight = model.experts
                # print(f"expert weight {expert_weight.norm()} gradexpert norm {expert_weight.grad.norm()}")
                for (k, v) in model.noisy_gating.gradient_cache.items():
                    wandb.log({k : v})

            aux_losses.append(aux_loss)
            losses.append(loss.item())

        model.eval()
        correct = 0 
        total = 0
        for (X_test, Y_test) in test_dataloader:
            if torch.cuda.is_available():
                X_test = X_test.to("cuda")
                Y_test = Y_test.to("cuda")
                model.to("cuda")
            if isinstance(model, ShazeerMOE):
                preds, _ = model.forward(X_test)
            else:
                preds = model.forward(X_test)
            assert preds.shape == (X_test.shape[0], D_out), f"shape of preds {preds.shape}, is not {X_test} * {D_out}"
            preds = preds.argmax(dim=-1)
            batch_correct = (preds == Y_test).sum()
            correct += batch_correct
            total += X_test.shape[0]

        model_name = {model.__class__.__name__}
        acc = correct / total 
        wandb.log({f"epoch" : i, f"acc" : acc, f"train_loss" : sum(losses) / len(losses), "aux_loss" : sum(aux_losses) / len(aux_losses)})
        print(f"Epoch {i} acc {acc}")


    plt.plot(range(len(losses)), losses)
    plt.savefig(f'{model_name}.loss_plot.png') 
    plt.clf()

def fit(moe, X, target_tensor, a = 0.01, num_epochs=2):
    if torch.cuda.is_available():
        moe.to("cuda")
    optim = torch.optim.Adam(moe.parameters(), lr=1e-3)

    losses = []
    for _ in range(num_epochs):
        optim.zero_grad()
        if torch.cuda.is_available():
            X = X.to("cuda")
            target_tensor = target_tensor.to("cuda")
        pred, aux_loss = moe.forward(X)
        loss = torch.nn.MSELoss()(pred, target_tensor)
        total_loss = loss + aux_loss * a

        total_loss.backward()
        optim.step()

        losses.append(loss.item())

    plt.plot(range(len(losses)), losses)
    plt.savefig('loss_plot.png') 
    plt.clf()
    


def test_non_zero_gradient():
    moe = ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H)

    X = torch.randn((B, S, D_in))
    Y = torch.randn((M, D_out))

    if torch.cuda.is_available():
        moe.to("cuda")
        X = X.to("cuda")
        Y = Y.to("cuda")

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
    models = [
        SimpleDNN(D=D_in,N=N, H=100),
        ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H),
    ]

    for model in models:
        train_dataloader, test_dataloader = gmm_dataset.generate_dataset(M, D_in, N, batch_size=1000)
        criterion = torch.nn.CrossEntropyLoss()
        model_name = model.__class__.__name__
        wandb.init(project=f"moe_benchmark", name=model_name, config=config, reinit=True)
        print(f"Model summary {model.__class__.__name__} {summary(model, input_size=(1, 2))}")
        wandb.watch(model, log="all", log_freq=100)
        fit_batch(model, train_dataloader, test_dataloader, criterion, D_out=N, num_epochs=25, a=0.25)


if __name__ == "__main__":
    B = 10000
    S = 5
    D_in = 2
    D_out = 4
    H = 100

    N = 4
    K = 3
    M = B * S
    config = {
        "B" : B,
        "H" : H,
        "N" : N,
        "K" : K 
    }


    test_non_zero_gradient()
    # test_synthetic_overfitting()
    # test_router_collapse()
    test_gmm_fit()
