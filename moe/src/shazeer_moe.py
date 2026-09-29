import torch 
import torch.nn as nn
import wandb
import math

class NoisyTopKGating(nn.Module):
    def __init__(self, D_in, N, K, magic=0):
        super(NoisyTopKGating, self).__init__()
        self.norm = torch.nn.LayerNorm(D_in)
        self.W_G = torch.nn.Parameter(torch.randn((D_in, N))/ math.sqrt(D_in))
        self.W_N = torch.nn.Parameter(torch.randn((D_in, N))/ math.sqrt(D_in))
        self.softplus = nn.Softplus()
        self.magic = magic

        self.D_in = D_in
        self.N = N 
        self.K = K

        self.register_buffer("loc", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.gradient_cache = {}

    @property
    def normal_dist(self):
        return torch.distributions.Normal(loc=self.loc, scale=self.scale)

    def forward(self, X:torch.tensor): # (B, S, D)
        X = self.norm(X)
        (B, S, D_in) = X.shape
        K, N = self.K, self.N

        W_G = X @ self.W_G # (B, S, D_in) @ (D_in, N) = (B, S, N)
        W_N = X @ self.W_N

        assert W_G.shape == (B, S, self.N)
        
        e = self.normal_dist.sample((B, S, self.N)) 

        H = W_G + e * self.softplus(W_N) # (B, S, N)
        if H.requires_grad:
            H.register_hook(lambda grad : self.gradient_cache.update({"H_grad" : grad.norm().item()}))
        KV, KI = torch.topk(H, k=self.K, dim=-1) # (B, S, K)

        assert KI.shape == (B, S, self.K)
        assert KV.shape == (B, S, self.K)

        G = nn.Softmax(dim=-1)(KV)

        assert G.shape == (B, S, K)

        if G.requires_grad:
            G.register_hook(lambda grad : self.gradient_cache.update({"G_grad" : grad.norm().item()}))

        # construct G_N (B, S, N) from G (B, S, K) and KI (B, S, N) where KI are indices. Torch.scatter will help here. 
        G_N = torch.zeros((B, S, N), dtype=G.dtype, device=G.device)
        G_N = G_N.scatter(dim=2, index=KI, src=G)

        assert G_N.shape == (B, S, N)

        probs = G_N.mean(dim=(0, 1))

        assert probs.shape == (self.N, )

        threshold_logit = KV[:,:,-1:]

        assert threshold_logit.shape == (B, S, 1)

        D = (W_G - threshold_logit) / self.softplus(W_N)
        assert D.shape == (B, S, self.N)

        D_clamped = torch.nan_to_num(torch.clamp(D, min=-10, max=10), posinf=10.0, neginf=-10.0)

        if D.requires_grad:
            D.register_hook(lambda grad : self.gradient_cache.update({"D_grad" : grad.norm().item()}))

        f = self.normal_dist.cdf(D_clamped).mean(dim=(0, 1))
        assert f.shape == (self.N, )
        if f.requires_grad:
            f.register_hook(lambda grad : self.gradient_cache.update({"F_grad" : grad.norm().item()}))

        aux_loss = torch.sum(f * probs)

        return G, aux_loss, KI     

class ShazeerMOE(nn.Module):
    def __init__(self, D_in, D_out, N, K, H):
        super(ShazeerMOE, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.N = N 
        self.K = K
        self.H = H
        
        self.experts = nn.Parameter(torch.randn((N, D_in, H), requires_grad=True))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(H, H),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            )

        self.noisy_gating = NoisyTopKGating(D_in, N, K)
        if torch.cuda.is_available():
            self.noisy_gating.to("cuda")
            self.experts.to("cuda")

    def forward(self, X):
        if len(X.shape) == 2:
            X = X.unsqueeze(dim=1)
        B, S, D_in = X.shape
        M = B * S
        K, N, D_out, H = self.K, self.N, self.D_out, self.H
        G, aux_loss, KI = self.noisy_gating(X)
        X = X.reshape((M, D_in))

        G = G.reshape((-1, K))

        assert G.shape == (M, K)

        KI = KI.reshape((-1, K))

        assert KI.shape == (M, K)

        EK = self.experts[KI]

        # N gets replaced with M, K 

        assert EK.shape == (M, K, D_in, H)

        XK = X.unsqueeze(dim=1).expand(-1, K, D_in)

        assert XK.shape == (M, K, D_in)

        Y = torch.einsum("mkd,mkde,mk->me", XK, EK, G)

        assert Y.shape == (M, H)

        return self.fc(torch.nn.ReLU()(Y)), aux_loss