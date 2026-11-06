### The Core Definition of $f$

In precise probabilistic terms, **$f_i$ is the expected probability that a given token will be routed to expert $i$**, accounting for the injected Gaussian noise. 

Averaged across all $M$ tokens in the batch, $f$ represents the **predicted fraction of total tokens** that will be assigned to each of the $N$ experts.

---

### The Formal Probability Notation

Let a token's raw routing score for expert $i$ be $W_{G,i}$, and let the $K$-th selection threshold (the score required to make the Top-$K$ cutoff) be $T$. 

We inject Gaussian noise $\epsilon_i \sim \mathcal{N}(0, 1)$ scaled by standard deviation $\sigma_i = \text{softplus}(W_{N,i})$, creating the noisy score:
$$H_i = W_{G,i} + \epsilon_i \sigma_i$$

A token selects expert $i$ if its noisy score exceeds the threshold $T$. The probability $f_i$ of this event is:
$$f_i = P(H_i > T) = P(W_{G,i} + \epsilon_i \sigma_i > T)$$

Rearranging this inequality to isolate the standard normal variable $\epsilon_i$:
$$f_i = P\left(\epsilon_i > \frac{T - W_{G,i}}{\sigma_i}\right)$$

Because the standard normal distribution is symmetric around zero ($P(\epsilon > -z) = P(\epsilon < z)$), this becomes the Cumulative Distribution Function ($\Phi$) of the standard normal distribution:
$$f_i = \Phi\left(\frac{W_{G,i} - T}{\sigma_i}\right)$$

Where:
*   $\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-\frac{t^2}{2}} dt$ is the standard normal CDF.
*   $\frac{W_{G,i} - T}{\sigma_i}$ is the Z-score ($D$).

---

### Batch Expectation

To compute the expected fraction for the entire batch of $M$ tokens, your code takes the mean across dimension $0$:

$$f_i = \frac{1}{M} \sum_{m=1}^{M} \Phi\left(\frac{W_{G,i}^{(m)} - T^{(m)}}{\sigma_i^{(m)}}\right)$$

This yields a vector $f \in \mathbb{R}^N$ representing the model's differentiable, soft estimate of expert utilization across the batch, which is then penalized via `aux_loss = torch.sum(f * probs)` to enforce load balancing.