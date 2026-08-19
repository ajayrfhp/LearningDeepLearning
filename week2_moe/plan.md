# 📅 1-Week Master Roadmap: Mixture of Experts (MoE)

## Overview
A comprehensive curriculum designed to build **PyTorch-source-code-level depth** across MoE architectures, covering routing mechanics, capacity limits, load balancing losses, tensor permutations, and distributed expert parallelism.

---

## Day 1: Mathematical Foundations & Gate Mechanics
**Objective:** Understand the fundamental difference between dense MLPs and sparse MoEs, and implement the router's score generation.

### Reading / Theory Tasks
1. Read Section 2 & 3 of *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (Shazeer et al.).
2. Derive the routing probability formula on paper:
   $$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$
   where $H(x)_i = (x \cdot W_g)_i + \epsilon \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$.
3. Compare computational complexity: $O(N \cdot d \cdot 4d)$ for a Dense FeedForward layer vs $O(k \cdot d \cdot 4d)$ per token in an MoE layer with $E$ experts ($k \ll E$).

### Coding Task (PyTorch)
* Write `NoisyTopKGating` class in PyTorch:
  * Inputs: Tensor $X \in \mathbb{R}^{B \times S \times D}$.
  * Computes linear projection $W_g \in \mathbb{R}^{D \times E}$.
  * Adds standard normal trainable noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ during training only.
  * Returns the top-$k$ routing weights and top-$k$ expert indices.

---

## Day 2: Top-k Routing, Capacity Factors, & Token Dropping
**Objective:** Master token dispatching and the concept of **Expert Capacity**.

### Reading / Theory Tasks
1. Read the capacity section of the *Switch Transformer Paper* (Fedus et al., 2021).
2. Study the Expert Capacity formula:
   $$\text{Capacity} = \left\lceil \frac{\text{Total Tokens}}{E} \times \text{Capacity Factor} \right\rceil$$
3. Understand why token dropping occurs when a specific expert receives more tokens than its allocated buffer, and how fallback/residual skip connections handle dropped tokens.

### Coding Task (PyTorch)
* Implement `build_expert_mask(indices, num_experts, capacity_factor)`:
  * Take top-$k$ indices of shape `(Batch * Seq_Len, k)`.
  * Calculate dynamic expert capacity based on batch size and `capacity_factor`.
  * Generate a binary tensor mask of shape `(num_experts, capacity, Batch * Seq_Len)` tracking which tokens get assigned to which expert slot.
  * Identify dropped tokens (tokens that exceeded capacity) and produce a boolean mask for residual routing.

---

## Day 3: Auxiliary Loss Engineering (Load Balancing & Z-Loss)
**Objective:** Master the exact math and code required to prevent **Router Collapse** (where the gate sends 99% of tokens to 1 or 2 experts).

### Reading / Theory Tasks
1. Read Section 3.2 of the *Switch Transformer Paper* on Load Balancing Loss.
2. Study the auxiliary loss equation:
   $$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i$$
   where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the fraction of total router probability allocated to expert $i$.
3. Read about **Router Z-Loss** (ST-MoE paper):
   $$\mathcal{L}_z = \frac{1}{B \cdot S} \sum_{j=1}^{B \cdot S} \left( \log \sum_{i=1}^E e^{z_{j,i}} \right)^2$$
   Understand how penalizing large unnormalized logits $z$ stabilizes FP16 training dynamics.

### Coding Task (PyTorch)
* Write `compute_moe_aux_losses(logits, expert_mask)`:
  * Compute $P_i$ (mean softmax probability per expert across the sequence).
  * Compute $f_i$ (percentage of active assignments per expert).
  * Return $\mathcal{L}_{\text{balance}}$ and $\mathcal{L}_z$ as scalar loss tensors ready for backpropagation.

---

## Day 4: PyTorch Source Code: Building `SparseMoE` Layer from Scratch
**Objective:** Assemble the components into a full end-to-end PyTorch module (`nn.Module`).

### Coding Task (PyTorch)
* Write the complete `SparseMoE` layer:
  ```python
  import torch
  import torch.nn as nn
  from typing import Tuple

  class SparseMoE(nn.Module):
      def __init__(
          self, 
          dim: int, 
          hidden_dim: int, 
          num_experts: int, 
          top_k: int = 2, 
          capacity_factor: float = 1.0
      ):
          super().__init__()
          # 1. Gate network
          # 2. ModuleList of N Experts (each an MLP: Linear -> GELU -> Linear)
          # 3. Load balancing loss tracker

      def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
          # x: (Batch, Seq_Len, Dim)
          # ...
          # Return (output_tensor, total_aux_loss)