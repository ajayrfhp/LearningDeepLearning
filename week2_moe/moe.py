import torch 
import torch.nn as nn

class NoisyTopKGating(nn.Module):
    def __init__(self, D, N, K):
        super(NoisyTopKGating, self).__init__()
        self.W_G = torch.nn.Parameter(torch.randn((D, N)))
        self.W_N = torch.nn.Parameter(torch.randn((D, N)))
        self.normal_dist = torch.distributions.Normal(loc=0, scale=1)
        self.softplus = nn.Softplus()

        self.D = D
        self.N = N 
        self.K = K


    def forward(self, X:torch.tensor): # (B, S, D)
        (B, S, D) = X.shape
        K, N = self.K, self.N
        W_G = X @ self.W_G # (B, S, D) @ (D, N) = (B, S, N)
        W_N = X @ self.W_N

        assert W_G.shape == (B, S, self.N)
        
        e = self.normal_dist.sample((B, S, self.N)) 

        H = W_G #+ e * self.softplus(W_N) # (B, S, N)
        KV, KI = torch.topk(H, k=self.K, dim=-1) # (B, S, K)

        assert KI.shape == (B, S, self.K)
        assert KV.shape == (B, S, self.K)


        G = nn.Softmax(dim=-1)(KV)

        assert G.shape == (B, S, K)

        # construct G_N (B, S, N) from G (B, S, K) and KI (B, S, N) where KI are indices. Torch.scatter will help here. 
        G_N = torch.zeros((B, S, N), dtype=G.dtype)
        G_N.scatter_(dim=2, index=KI, src=G)

        assert G_N.shape == (B, S, N)

        probs = G_N.mean(dim=(0, 1))

        assert probs.shape == (self.N, )

        threshold_logit = KV[:,:,-1:]

        assert threshold_logit.shape == (B, S, 1)

        D = (W_G - threshold_logit) / self.softplus(W_N)

        assert D.shape == (B, S, self.N)

        f = self.normal_dist.cdf(D).mean(dim=(0, 1))

        assert f.shape == (self.N, )

        aux_loss = torch.sum(f * probs)

        return G, aux_loss, KI     


class Block(nn.Module):
    def __init__(self, D):
        super(Block, self).__init__()
        self.a1 = nn.Linear(D, D)
        self.a2 = nn.Linear(D, D)

    def forward(self, x):
        h1 = self.a1(x)
        h2 = nn.GELU()(h1)
        return self.a2(h2)

class ShazeerMOE(nn.Module):
    def __init__(self, D, N, K):
        super(ShazeerMOE, self).__init__()
        self.D = D
        self.N = N 
        self.K = K
        
        self.experts = nn.Parameter(torch.randn((N, D, D), requires_grad=True))

        self.noisy_gating = NoisyTopKGating(D, N, K)

    def forward(self, X):
        B, S, D = X.shape
        M = B * S
        K, N = self.K, self.N

        G, aux_loss, KI = self.noisy_gating(X)
        X = X.reshape((M, D))

        G = G.reshape((-1, K))

        assert G.shape == (M, K)

        KI = KI.reshape((-1, K))

        assert KI.shape == (M, K)

        EK = self.experts[KI]

        # N gets replaced with M, K 

        assert EK.shape == (M, K, D, D)

        XK = X.unsqueeze(dim=1).expand(-1, K, D)

        assert XK.shape == (M, K, D)

        Y = torch.einsum("mkd,mkde,mk->me", XK, EK, G)

        assert Y.shape == (M, D)

        return Y, aux_loss