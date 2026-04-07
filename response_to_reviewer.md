$\def \Rb {\mathbb{R}}$
$\def \Eb {\mathbb{E}}$
$\def \tr  {\mathrm{true}}$
$\def \tn {\mathrm{noise}}$

Dear AC,

Thanks for your time and consideration. Since the Reviewer yaeN was added after the rebuttal period, we did not have an opportunity to respond in the public discussion. We would therefore be very grateful if you could kindly forward the following response to the review on our behalf.

## On theoretical analysis of AGOP in MIPL

The key difficulty in MIPL is not only candidate-label ambiguity, but also that the true bag label is usually supported by only a small subset of key instances, while many irrelevant instances remain in the bag. AGOP helps because it does not directly denoise labels, instead, it reshapes the feature geometry according to the sensitivity of the bag-level predictor. Concretely, we compute AGOP on the bag-level representation $z \in \Rb^{d'}$:
$$
G = \Eb\left[J_f(z)^\top J_f(z)\right],
$$
where $J_f(z)$ is the Jacobian of the bag-level predictor with respect to $z$. Thus, AGOP emphasizes directions in representation space along which the predictor is consistently sensitive across training bags.

A useful stylized interpretation is to decompose the bag-level gradient into a task-relevant part and a noise-induced part:
$$g = g_{\tr} + g_{\tn}.$$
If
$$
\Eb[g_{\tr} g_{\tr}^\top] = U \Lambda U^\top,\quad
\Eb[g_{\tn} g_{\tn}^\top] = \sigma^2 I,\quad
\Eb[g_{\tr} g_{\tn}^\top] = 0,
$$
Where $g_{\tr}$ denotes the component aligned with the true-label / key-instance signal, while $g_{\tn}$ denotes the component induced by false-positive candidate labels and irrelevant instances. Then
$$
G=\Eb[g g^\top] = U\Lambda U^\top + \sigma^2 I.
$$
Hence the top-$r$ eigenspace of AGOP is exactly the discriminative subspace $U$. Moreover, if $x=x_U+x_\perp$ with $x_U\in U$ and $x_\perp\in U^\perp$, then after the AGOP transform $G^{1/2}$,
$$
\frac{\|G^{1/2}x_U\|}{\|G^{1/2}x_\perp\|}
\ge
\sqrt{1+\frac{\lambda_r}{\sigma^2}}
\frac{\|x_U\|}{\|x_\perp\|}.
$$
This provides a mechanism for why AGOP is useful in noisy MIPL: directions related to the true label are more coherent across bags, whereas false-positive-label directions are less consistent, so AGOP improves key-instance separation before attention aggregation.

## On the scalability of our method

In our implementation, AGOP is computed on the **bag-level representation** $z\in\mathbb{R}^{d'}$, the practical overhead is governed mainly by $d'$, the number of training bags, and the AGOP update frequency.

We first profiled scalability with respect to dataset size at fixed $d'=128$:

| Dataset | Train Bags | $d'$ | Best Acc | Round-1 Train (s) | Round-1 AGOP (s) | AGOP Overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FMNIST-MIPL ($r=3$) | 250 | 128 | 0.712 | 133.53 | 0.42 | 0.31% |
| SIVAL-MIPL ($r=3$) | 750 | 128 | 0.680 | 546.38 | 1.37 | 0.25% |
| CRC-MIPL-KMeansSeg | 4900 | 128 | 0.706 | 1970.02 | 3.40 | 0.17% |

These results show that although the update time increases with the number of training bags, the AGOP-specific overhead remains very small relative to one training round.

We next profiled scalability with respect to the bag-level feature dimension $d'$ on CRC-MIPL-KMeansSeg (fold 1):

| $d'$ | Best Acc | Round-1 Train (s) | Round-1 AGOP (s) | AGOP Overhead | AGOP Peak Mem (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64  | 0.703 | 1972.23 | 3.42 | 0.17% | 17.69 |
| 128 | 0.706 | 1970.02 | 3.40 | 0.17% | 19.05 |
| 256 | 0.714 | 2021.05 | 3.61 | 0.18% | 22.63 |
| 512 | 0.705 | 2047.95 | 3.77 | 0.18% | 30.28 |

This shows that increasing $d'$ from 64 to 512 keeps the performance essentially stable, while AGOP update time and memory increase only mildly.

Regarding transformer encoders, prior AGOP work is not restricted to CNN/MLP architectures and has also studied transformer-based models. To directly test feasibility in our pipeline, we replaced the default instance encoder with a lightweight transformer encoder (2 layers, 4 heads) while still applying AGOP at the projected bag embedding level. On SIVAL-MIPL, the transformer-based variant remained trainable and achieved **0.651** test accuracy, compared with **0.680** for the default encoder. The AGOP-specific overhead also remained small relative to the full training time:

| Encoder | Dataset | Train Bags | $d'$ | Best Acc | Round-1 Train (s) | Round-1 AGOP (s) | AGOP Overhead | AGOP Peak Mem (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default encoder | SIVAL-MIPL ($r=3$) | 750 | 128 | 0.680 | 546.60 | 1.36 | 0.25% | 23.76 |
| Lightweight Transformer | SIVAL-MIPL ($r=3$) | 750 | 128 | 0.651 | 3376.44 | 7.60 | 0.23% | 29.81 |

Overall, these new results shows that AGOP is particularly useful in noisy MIPL because it amplifies stable discriminative directions before attention aggregation, and AGOP-specific overhead remains small even as dataset size and $d'$ increase. Moreover, our method is feasible for transformer-based encoders.
