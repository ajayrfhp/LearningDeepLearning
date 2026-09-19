## 1. Overview 

## 2. Releases
- v0.1 Shazeer MoE & Math foundations
- v0.2 Switch transformer 
- v0.3 ?

## 3. Shazeer MoE
### Overview
Idea is to keep only part of a large DNN active for a given input and having the DNN figure out which parts to keep active using a gate. The network is composed of N experts out of which only K is kept active for a given set of inputs. 

Given token representation $x \in \mathbb{R}^D$:
$$H(x)_i = (x \cdot W_{\text{gate}})_i + \epsilon \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i), \quad \epsilon \sim \mathcal{N}(0, 1)$$

$W_{gate}$ is the learnt gating layer that learns mapping over which set of experts attend to. 
$W_{noise}$ is a noise injector that prevents a single expert that learns early to dominate gradient flows and become the only expert that learns. 

From the set of N experts, the topK function picks the topk expert activations to keep and applies softmax. 

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

Standard learning loss is enhanced with an auxillary load balancing loss which ensures routers learn equal stuff and are not overloaded. 

* **Router Probability Fraction ($P_i$):** Mean routing probability allocated to expert $i$ across sequence length $N = B \cdot S$.
  $$P_i = \frac{1}{N} \sum_{x \in X} G(x)_i \quad \text{(Shazeer Importance Metric)}$$

Token assignment fraction / load balancing loss ensures all experts recieve good chunk of tokens ensuring gpu utilization is uniform. 
* **Token Assignment Fraction ($f_i$):** Fraction of tokens routed to expert $i$.
  * **Shazeer (Differentiable Soft Load):** $f_i = \frac{1}{N} \sum_{x \in X} P(x, i)$, where $P(x, i) = \Phi\left(\frac{(x \cdot W_{\text{gate}})_i - \text{Threshold}(x)}{\sigma_i(x)}\right)$ using standard Gaussian CDF $\Phi$.

Losses are combined

* **Shazeer CDF Loss:** $\mathcal{L}_{\text{aux}} = w_{\text{imp}} \cdot \text{CV}(P)^2 + w_{\text{load}} \cdot \text{CV}(f)^2$ (where $\text{CV}(v) = \frac{\sigma(v)}{\mu(v)}$).

### Implementation 

### Testing
