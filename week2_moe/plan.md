# Shazeer MoE Master Plan (Updated with $f_i, P_i$ Mapping & Fast Verification)

## Phase 1: Mathematical Foundations & Loss Definitions

### 1. Noisy Gating Score
Given token representation $x \in \mathbb{R}^D$:
$$H(x)_i = (x \cdot W_{\text{gate}})_i + \epsilon \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i), \quad \epsilon \sim \mathcal{N}(0, 1)$$

### 2. Top-$k$ Softmax Selection
$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

### 3. $f_i$ and $P_i$ Auxiliary Loss Mapping
* **Router Probability Fraction ($P_i$):** Mean routing probability allocated to expert $i$ across sequence length $N = B \cdot S$.
  $$P_i = \frac{1}{N} \sum_{x \in X} G(x)_i \quad \text{(Shazeer Importance Metric)}$$
* **Token Assignment Fraction ($f_i$):** Fraction of tokens routed to expert $i$.
  * **Shazeer (Differentiable Soft Load):** $f_i = \frac{1}{N} \sum_{x \in X} P(x, i)$, where $P(x, i) = \Phi\left(\frac{(x \cdot W_{\text{gate}})_i - \text{Threshold}(x)}{\sigma_i(x)}\right)$ using standard Gaussian CDF $\Phi$.
  * **Switch/GShard (Hard Discrete Load):** $f_i = \frac{1}{N \cdot k} \sum_{x \in X} \mathbb{I}\left(\text{expert } i \in \text{TopK}(x)\right)$.
* **Loss Formulas:**
  * **Shazeer CDF Loss:** $\mathcal{L}_{\text{aux}} = w_{\text{imp}} \cdot \text{CV}(P)^2 + w_{\text{load}} \cdot \text{CV}(f)^2$ (where $\text{CV}(v) = \frac{\sigma(v)}{\mu(v)}$).
  * **Switch / GShard Loss:** $\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$.

---

## Phase 2: PyTorch Source Code & Module Architecture

### Step 1: `NoisyTopKGating` Module
1. Project $X \in \mathbb{R}^{B \times S \times D} \to W_{\text{gate}}(X)$ and $W_{\text{noise}}(X)$.
2. Sample standard normal noise $\epsilon$ during `training` mode.
3. Compute Top-$k$ indices via `torch.topk`.
4. Apply Softmax strictly over the top $k$ selected logits.

### Step 2: Auxiliary Loss Calculation (`compute_moe_aux_losses`) — [Where $f_i, P_i$ live]
1. Compute **$P_i$**: Average Softmax probabilities $G(x)_i$ across batch and sequence dimensions ($B \cdot S$).
2. Compute **$f_i$**:
   * For Shazeer: Evaluate Gaussian CDF $\Phi\left(\frac{W_{\text{gate}}(x)_i - \text{Threshold}(x)}{\text{Softplus}(W_{\text{noise}}(x)_i)}\right)$ per token and take the mean across $B \cdot S$.
   * For Switch: Count active Top-$k$ expert assignments per expert and normalize by total assignments ($B \cdot S \cdot k$).
3. Compute Coefficient of Variation $\text{CV}(P)$ and $\text{CV}(f)$ (or dot product $E \sum f_i P_i$).
4. Return scalar loss terms $\mathcal{L}_{\text{imp}}$ and $\mathcal{L}_{\text{load}}$ (or $\mathcal{L}_{\text{balance}}$).

### Step 3: `ShazeerMoE` Layer Assembly
1. Instantiate `nn.ModuleList` of $E$ independent MLPs (`Linear` $\to$ `GELU` $\to$ `Linear`).
2. Dispatch active tokens to assigned experts.
3. Combine outputs: $y = \sum_{i \in \text{TopK}} G(x)_i \cdot \text{Expert}_i(x)$.
4. Return output tensor `(B, S, D)` and scalar `total_aux_loss`.

---

## Phase 3: Fast & Cheap Verification Strategy ($0 Spent, <2 Minutes on CPU/Single GPU)

To test if the model learns, avoids router collapse, and spreads auxiliary loss without expensive multi-GPU runs:

### Test 1: Autograd & Gradient Flow Smoke Test (Execution time: ~2 seconds)
* Pass synthetic tensor $X \in \mathbb{R}^{2 \times 4 \times 16}$ through `ShazeerMoE`.
* Execute `loss.backward()`.
* **Assertion:** Assert `.grad` is non-None and non-zero for $W_{\text{gate}}$, $W_{\text{noise}}$, and active expert parameters.

### Test 2: Synthetic Overfitting / Learning Capability Test (Execution time: ~15 seconds)
* Create a toy classification dataset (64 synthetic samples, 4 classes, random clusters).
* Train a single-layer `ShazeerMoE` for 50 steps using AdamW ($lr=1e-3$).
* **Assertion:** Assert training loss drops below $0.05$, proving the sparse routing pipeline can optimize weights.

### Test 3: Anti-Router Collapse & Aux Loss Stress Test (Execution time: ~30 seconds)
* Initialize $W_{\text{gate}}$ with heavy manual bias toward Expert 0 ($W_{\text{gate}}[0] = +10.0$) to simulate instant router collapse.
* Run two parallel 50-step synthetic training loops:
  * **Run A:** Aux Loss multiplier $w_{\text{aux}} = 0.0$.
  * **Run B:** Aux Loss multiplier $w_{\text{aux}} = 0.01$.
* Track expert assignment fractions $f_i$ across steps.
* **Assertion:** In Run A, $f_0 \approx 1.0$ (collapse persists). In Run B, $\text{CV}(f)$ decreases significantly and tokens redistribute across all $E$ experts ($f_i \to \frac{1}{E}$).

### Test 4: Uniform Input Symmetry Test (Execution time: ~2 seconds)
* Pass zero input $X = \mathbf{0}$ or uniform inputs with zero noise $\epsilon = 0$.
* **Assertion:** Assert $P_i = \frac{1}{E}$ and $f_i = \frac{1}{E}$ across all experts within floating-point tolerance (`torch.allclose`).