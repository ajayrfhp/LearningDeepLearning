import torch
import sys, os
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../src")

from shazeer_moe import NoisyTopKGating, ShazeerMOE
from simple_dnn import SimpleDNN
import gmm_dataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torchinfo import summary
import wandb 
from torch.utils.data import TensorDataset, DataLoader

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

            assert pred.is_floating_point(), "pred is not float"
            assert target_tensor.is_floating_point(), "target is not float"
            
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
                expert_up_weight = model.experts_up
                expert_down_weight = model.experts_down
                # print(f"expert weight {expert_weight.norm()} gradexpert norm {expert_weight.grad.norm()}")
                for (k, v) in model.noisy_gating.gradient_cache.items():
                    wandb.log({k : v})

                expert_norm = expert_up_weight.abs().mean(dim=(1, 2)).detach().cpu().numpy()
                assert len(expert_norm) == model.N
                wandb.log({f"expert_norms/expert_up_{i}": val for i, val in enumerate(expert_norm)})

                wandb.log({
                    "metrics/expert_up_norm_std": np.std(expert_norm),
                    "metrics/expert_up_norm_max_min_diff": np.max(expert_norm) - np.min(expert_norm)
                })

            aux_losses.append(a * aux_loss)
            losses.append(loss.item())

        model.eval()
        correct = 0 
        total = 0
        acc = 0
        test_losses = []
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
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                preds = preds.argmax(dim=-1)
                batch_correct = (preds == Y_test).sum()
                correct += batch_correct
                total += X_test.shape[0]

            if isinstance(criterion, torch.nn.MSELoss):
                test_loss = criterion(preds, target_tensor)
                test_losses.append(test_loss.item())

        model_name = {model.__class__.__name__}
        if total > 0:
            acc = correct / total 
        wandb.log({f"epoch" : i, f"acc" : acc, f"train_loss" : sum(losses) / len(losses), "aux_loss" : sum(aux_losses) / len(aux_losses), "test_loss" : sum(test_losses) / len(test_losses)})
        print(f"Epoch {i} acc {acc}")


def test_non_zero_gradient():
    B = 100
    S = 5
    D_in = 2
    D_out = 4
    H = 100
    num_epochs = 50

    N = 4
    K = 3
    M = B * S
    config = {
        "B" : B,
        "H" : H,
        "N" : N,
        "K" : K,
        "num_epochs" : num_epochs
    }
    test_size = 0.8
    batch_size = 100

    moe = ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H)
    X = torch.randn((M, D_in))
    Y = torch.randn((M, D_out))

    if torch.cuda.is_available():
        moe.to("cuda")
        X = X.to("cuda")
        Y = Y.to("cuda")

    indices = torch.randperm(X.shape[0])
    X, Y = X[indices], Y[indices]

    train_length = int(test_size * X.shape[0])
    X_train, X_test, Y_train, Y_test = X[:train_length], X[train_length:], Y[:train_length], Y[train_length:]
    train_dataset = TensorDataset(X_train.to(torch.float), Y_train.to(torch.float))
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test.to(torch.float), Y_test.to(torch.float))
    test_dataloader = DataLoader(test_dataset, batch_size)
    criterion = torch.nn.MSELoss()

    model_name = f"non_zero_gradient"
    wandb.init(project=f"moe_benchmark_test", name=model_name, config=config, reinit=True)
    wandb.watch(moe, log="all", log_freq=100)
    fit_batch(moe, train_dataloader, test_dataloader, criterion, D_out=N, num_epochs=num_epochs, a=0)

    # grad should be non zero for W_G, W_N and W_EK

    gradients = {item[0] : item[1].grad for item in moe.named_parameters() if item[1].grad is not None}

    w_g = gradients['noisy_gating.W_G']
    w_n = gradients['noisy_gating.W_N']
    w_e_up = gradients['experts_up']
    w_e_down = gradients['experts_down']

    assert w_g.sum().abs() > 0 and w_n.sum().abs() > 0 and w_e_up.sum().abs() > 0 and w_e_down.sum().abs() > 0

def test_synthetic_overfitting():
    B = 100
    S = 5
    D_in = 2
    D_out = 4
    H = 100
    num_epochs = 50

    N = 4
    K = 3
    M = B * S
    config = {
        "B" : B,
        "H" : H,
        "N" : N,
        "K" : K,
        "num_epochs" : num_epochs
    }
    test_size = 0.8
    batch_size = 100

    X = torch.randn((M, 2))
    Y = (X.clone().detach())  * 3 
    # Y = Y.unsqueeze(-1).expand(-1, -1, D)
    assert Y.shape == (M, 2)

    moe = ShazeerMOE(D_in=2, D_out=2, N=N, K=K, H=H)
    fit_batch(moe, X, Y, num_epochs=100, a = 0)

    new_inpt = torch.tensor([[1.0, 1.0]])
    new_inpt = new_inpt.reshape((-1, 2))
    expected_gt = torch.tensor([[3.0, 3.0]]).reshape((-1, 2))
    if torch.cuda.is_available():
        moe.to("cuda")
        X = X.to("cuda")
        Y = Y.to("cuda")
        new_inpt = new_inpt.to("cuda")
        expected_gt = expected_gt.to("cuda")

    pred, _ = moe.forward(new_inpt)

    assert torch.allclose(pred[0], expected_gt, atol=0.5), f"{pred} is not 3"

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
    B = 10000
    S = 5
    D_in = 2
    D_out = 4
    H = 100
    num_epochs = 50

    N = 4
    K = 3
    M = B * S
    config = {
        "B" : B,
        "H" : H,
        "N" : N,
        "K" : K,
        "num_epochs" : num_epochs
    }

    models = [
        #(None, SimpleDNN(D=D_in,N=N, H=100)),
        (0, (ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H, noise_penalty=0))),
        (0.01, (ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H, noise_penalty=1))),
        (0.02, (ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H, noise_penalty=1))),
        #(0.1, (ShazeerMOE(D_in=D_in, D_out=N, N=N, K=K, H=H)))
    ]

    for a, model in models:
        train_dataloader, test_dataloader = gmm_dataset.generate_dataset(M, D_in, N, batch_size=1000)
        criterion = torch.nn.CrossEntropyLoss()
        model_name = f"{model.__class__.__name__}_{a}"
        wandb.init(project=f"moe_benchmark", name=model_name, config=config, reinit=True)
        print(f"Model summary {model.__class__.__name__} {summary(model, input_size=(1, 2))}")
        wandb.watch(model, log="all", log_freq=100)
        fit_batch(model, train_dataloader, test_dataloader, criterion, D_out=N, num_epochs=num_epochs, a=a)


if __name__ == "__main__":


    test_non_zero_gradient()
    # test_synthetic_overfitting()
    # test_router_collapse()
    # test_gmm_fit()
