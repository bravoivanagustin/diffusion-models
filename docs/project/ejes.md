# Diffusion Model Project — Two Experimental Axes

## Overview

The core idea is a **controlled ablation study**: keep the neural network architecture fixed and vary only the *stochastic* components of the diffusion framework. This isolates the effect of the mathematical choices from the engineering choices, which is exactly what matters for a stochastic calculus course.

The project proceeds in **two phases**:

**Phase 1 — 2D toy distributions with an MLP.** Train on low-dimensional synthetic data (Swiss roll, Gaussian mixtures, concentric circles). The score network is a small multi-layer perceptron built from scratch. This phase prioritizes theoretical depth: you can visualize score fields, particle trajectories, and density evolution directly in the plane, and for Gaussian mixtures you can compare the learned score against the analytically computed true score. Training takes minutes on CPU. This phase produces the theoretical core of the monograph and the most visually informative figures for the poster.

**Phase 2 — Image data with a U-Net.** Scale the same SDE framework to FashionMNIST or CIFAR-10, replacing the MLP with a standard U-Net from an existing library (e.g., HuggingFace `diffusers` or Lucidrains' `denoising-diffusion-pytorch`). This phase demonstrates that the theory generalizes to real high-dimensional data and produces FID/IS benchmarks. Training takes 12–20 hours per run on a single GPU (RTX 3060).

In both phases, the neural network is the **control variable** — the architecture and hyperparameters stay identical across all runs. The **independent variables** are the forward SDE type (Axis 1) and the reverse-process sampler (Axis 2). The **dependent variables** are sample quality, convergence behavior, and visual characteristics.

The study has two independent axes that combine into a matrix of experiments.

---

## The Score Network: Architecture and Its Role in the Project

The score network approximates $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ — the gradient of the log-density of the noised data at time $t$. It is a **deterministic** function: given the same input $(x_t, t)$, it always produces the same output. There are no random layers, no sampling operations, no latent variables inside the network. All the stochasticity in a diffusion model lives in the SDE framework surrounding the network.

The project uses two different architectures for the two phases, chosen to match the dimensionality of the data.

### Phase 1: MLP (for 2D toy data)

When the data is just 2D points, there are no spatial dimensions to exploit — convolutions and downsampling would be meaningless. Instead, the score network is a simple **multi-layer perceptron (MLP)**: a stack of fully connected layers with nonlinear activations.

A typical configuration for this project:

- **Input:** a 2D point $(x, y)$ concatenated with the timestep embedding → a vector of dimension $2 + d_\text{emb}$.
- **Hidden layers:** 4 residual blocks of 256 units each, with SiLU (or ReLU) activations.
- **Output:** a 2D vector — the predicted score $\nabla_x \log p_t(x)$ at the input point and noise level.

Residual connections (skip connections within the MLP, adding each block's input to its output) stabilize training even in this small network.

This MLP is built from scratch in PyTorch — roughly 30–40 lines of code. No external library is needed.

```
Input: (x, y) ∈ ℝ²    Timestep: t ∈ [0, T]
       │                        │
       │                 [Sinusoidal Embedding]
       │                        │
       │                        ▼
       │                  t_emb ∈ ℝ^d
       │                        │
       └────────► concat ◄──────┘
                    │
                    ├───────────┐
                    ▼           │ residual
            [Linear + SiLU]     │
                    │           │
                    + ◄─────────┘
                    │
                    ├───────────┐
                    ▼           │ residual
            [Linear + SiLU]     │
                    │           │
                    + ◄─────────┘
                    │
                    ├───────────┐
                    ▼           │ residual
            [Linear + SiLU]     │
                    │           │
                    + ◄─────────┘
                    │
                    ├───────────┐
                    ▼           │ residual
            [Linear + SiLU]     │
                    │           │
                    + ◄─────────┘
                    │
                    ▼
              [Linear → ℝ²]
                    │
                    ▼
         Output: predicted score ∈ ℝ²
```

### Phase 2: U-Net (for image data)

When the data consists of images, the spatial structure matters — neighboring pixels are correlated, and features exist at multiple scales. The **U-Net** is a convolutional neural network originally introduced by Ronneberger, Fischer & Brox (2015) for biomedical image segmentation, later adapted for diffusion models by Ho et al. (2020). Its defining feature is an hourglass (encoder–decoder) shape with skip connections:

1. **Encoder (downsampling path):** A series of convolutional blocks, each followed by a spatial downsampling operation (e.g., strided convolution or max-pooling). Each block doubles the number of feature channels while halving the spatial resolution. This path compresses the input into a compact, high-level representation — it captures *what* is in the image at the expense of *where*.

2. **Bottleneck:** The deepest, lowest-resolution layer. Here the network has the most abstract representation of the input.

3. **Decoder (upsampling path):** Mirrors the encoder — each block upsamples the spatial resolution and halves the number of channels, progressively reconstructing a full-resolution output.

4. **Skip connections:** At each resolution level, the encoder's output is concatenated (or added) to the corresponding decoder layer. The decoder gets both the high-level semantic information flowing up from the bottleneck *and* the fine-grained spatial details carried over from the encoder. Without skip connections, the network would lose precise localization information through the bottleneck.

Schematically, data flows down the left side (encoder), turns through the bottleneck at the bottom, and climbs back up the right side (decoder) — tracing a U shape:

```
Input (32×32×3)                                       Output (32×32×3)
  │                                                          ▲
  ▼                                                          │
[Conv Block] ─────────────────────── skip ───► [Conv Block + Upsample]
  │                                                          ▲
  ▼                                                          │
[Conv Block + Downsample] ──────── skip ───► [Conv Block + Upsample]
  │                                                          ▲
  ▼                                                          │
[Conv Block + Downsample] ──────── skip ───► [Conv Block + Upsample]
  │                                                          ▲
  ▼                                                          │
  └─────────────────► [Bottleneck] ──────────────────────────┘
```

Ho et al. (2020) adapted this architecture for DDPM with several additions: **timestep conditioning** (see below), **self-attention layers** at intermediate resolutions (commonly 16×16) for long-range spatial dependencies, **group normalization** instead of batch normalization for small-batch stability, and **residual connections** within each convolutional block.

For this project, the U-Net is taken from an existing library (HuggingFace `diffusers` or Lucidrains' `denoising-diffusion-pytorch`) rather than built from scratch — the SDE framework around the network is where the original contribution lives.

### Timestep embedding (shared by both architectures)

The score function depends on both the data $x$ and the noise level $t$: the network must know *how noisy* its input is to adjust its behavior. At low noise the network makes fine corrections; at high noise it makes large structural moves.

Simply concatenating the raw scalar $t$ works poorly — a single number gives the network too little to learn complex nonlinear dependencies from. Instead, **sinusoidal positional embeddings** expand $t$ into a high-dimensional vector using alternating sine and cosine functions at different frequencies:

$$\text{embed}(t)_{2i} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{embed}(t)_{2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

If $d = 128$, this produces 64 sines and 64 cosines, each oscillating at a different rate. Low-frequency components capture coarse distinctions (early vs. late in the process); high-frequency components capture fine distinctions between nearby timesteps. Two nearby values (t = 0.50 and t = 0.51) get similar but distinguishable embeddings, while distant values (t = 0.1 and t = 0.9) get very different ones.

How this embedding enters the network differs by architecture:

- **In the MLP** (Phase 1): the embedding vector is concatenated with the 2D input to form a $(2 + d)$-dimensional vector fed into the first layer.
- **In the U-Net** (Phase 2): the embedding passes through a learned linear layer and is injected into each residual block via an affine transformation — it produces a scale $\gamma$ and shift $\beta$ that modulate features after normalization, as in $x \mapsto \gamma \cdot \text{normalize}(x) + \beta$. This way every layer of the network is informed about the current noise level.

### The network is the control variable — stochasticity lives outside it

In a well-designed experiment, you change one thing at a time. In this project:

- **Independent variables** (what we vary): the forward SDE type (Axis 1) and the reverse sampler (Axis 2).
- **Control variable** (what we hold fixed): the network architecture — same depth, same widths, same learning rate, same optimizer, same number of training steps.
- **Dependent variables** (what we measure): sample quality, convergence speed, visual characteristics.

All the stochasticity in a diffusion model lives in the SDE framework surrounding the network:

- The **forward process** (stochastic): the Itô SDE that adds noise to data.
- The **training pairs** (stochastic): at each training step, a random $t$ is sampled and random noise is drawn to create $x_t$ from $x_0$.
- The **reverse sampler** (stochastic or deterministic depending on choice): Euler–Maruyama injects noise; the probability-flow ODE does not.

By freezing the architecture, every measurable difference in output is guaranteed to come from the SDE framework — which is exactly what we want to study in a stochastic calculus project.

### Parameterization choice: score prediction

The network's output can be framed in three mathematically equivalent ways:

- **Noise prediction:** output $\hat{\epsilon}$ (used by DDPM).
- **Score prediction:** output $\hat{s}(x, t) \approx \nabla_x \log p_t(x)$ (used by Song et al. 2021).
- **Clean data prediction:** output $\hat{x}_0$ (used by some implementations).

Given any one, you can compute the other two via a known rescaling. For this project we use **score prediction**, since the score $\nabla_x \log p_t(x)$ is the quantity that appears directly in the reverse SDE and the probability-flow ODE — keeping the code aligned with the mathematics in the monograph.

---

## 2D Toy Datasets (Phase 1)

Phase 1 uses synthetic 2D distributions where each data sample is a point $(x, y) \in \mathbb{R}^2$. The model generates individual 2D coordinates — not images. Visualizations in papers showing Swiss rolls or Gaussian blobs are scatter plots of thousands of such generated points.

Three datasets are used, each testing a different property of the SDE variants:

### Swiss roll

A spiral that curls around itself roughly two full turns. The data lies on a one-dimensional manifold embedded in 2D, so the score must follow curvature rather than just pointing toward a center. Different forward SDEs unwind the spiral in visually distinct ways — VP's drift shrinks it toward the origin, while VE buries it in place under noise. Produces the most striking trajectory visualizations for the poster.

### 8-Gaussian mixture

Eight Gaussian blobs with small variance arranged evenly on a circle. The gaps between modes are the challenge: the score must be nearly zero in dead zones and strongly directional near each cluster. This is the best dataset for testing **mode coverage** (if the model drops a mode, you see an empty cluster immediately) and the only one where the **true analytical score** can be computed for comparison — a Gaussian mixture convolved with Gaussian noise remains a Gaussian mixture.

### Two concentric circles

An inner ring at radius $r_1$ and an outer ring at radius $r_2$, both with slight radial noise. Tests whether the reverse process respects the topological gap — generated points should fall *on* the rings, not between them. Visually clean and immediately interpretable.

All three are one-liners to generate with scikit-learn (`make_swiss_roll`, `make_blobs`, `make_circles`).

### Why 2D toy data matters

For the stochastic calculus narrative, 2D data is more valuable than images because:

- You can **visualize the learned score field** as a vector field overlaid on the density — arrows pointing "downhill" toward high-probability regions.
- You can **plot forward and reverse SDE trajectories** as particle paths in the plane, directly showing how different SDEs move data through space.
- You can **compute the true score analytically** for Gaussian mixtures and compare it against the network's approximation.
- Training takes minutes on CPU, enabling fast iteration.

None of this is possible with CIFAR-10, where the data lives in ~3000-dimensional space.

---

## Why Each Forward SDE Requires Training from Scratch

A trained score network approximates $s_\theta(x, t) \approx \nabla_x \log p_t(x)$, where $p_t(x)$ is the distribution of data after being noised for time $t$ under a *specific* forward SDE. The key insight is that **$p_t(x)$ is different for every forward SDE**, even at the same timestep $t$.

For example, at some intermediate $t$:

- Under **VP-SDE**, the data has been shrunk toward zero *and* noise has been added — $p_t$ is concentrated, roughly Gaussian, moderate variance.
- Under **VE-SDE**, the data hasn't been shrunk at all, just buried under growing noise — $p_t$ is much wider, with a different shape.
- Under **CLD**, the distribution lives in the *joint* $(x, v)$ space and has correlations between position and momentum that don't exist in the other SDEs.

Since the score $\nabla_x \log p_t(x)$ is the gradient of a *different* density in each case, a network trained for one SDE has learned the wrong function for any other SDE. The weights are not transferable.

However, within **Axis 2** (varying the reverse sampler), no retraining is needed. The reverse-time SDE and the probability-flow ODE share the *same* score function — they differ only in whether and how noise is injected during sampling. So a single trained model supports all four samplers. This is why Axis 2 is computationally free: you're reusing the same learned score with different numerical integration schemes.

**Summary of training requirements:**

| Experiment | Retraining needed? | Why? |
|---|---|---|
| Changing the forward SDE (Axis 1) | **Yes** — one full training run per SDE variant | Different forward SDE → different $p_t(x)$ → different score target |
| Changing the reverse sampler (Axis 2) | **No** — swap the sampling loop only | All samplers use the same learned score $s_\theta(x,t)$ |

**Training time estimates:**

| Phase | Dataset | Network | Time per run | Total (4 SDEs) |
|---|---|---|---|---|
| Phase 1 | 2D toy data | MLP | ~2–5 minutes (CPU) | ~10–20 minutes |
| Phase 2 | CIFAR-10 / FashionMNIST | U-Net | ~12–20 hours (RTX 3060) | ~2–4 days |

---

## Axis 1 — Forward SDE Type (requires retraining)

Each variant defines a different stochastic differential equation that gradually destroys the data distribution. Changing the forward SDE changes everything downstream: the reverse-time SDE (via Anderson's theorem), the score-matching training objective, and the sampling dynamics.

### 1.1 VP-SDE (Variance Preserving)

$$dX_t = -\tfrac{1}{2}\beta(t)\,X_t\,dt + \sqrt{\beta(t)}\,dW_t$$

- **Origin:** Continuous-time limit of DDPM (Ho et al., 2020).
- **Behavior:** The drift $-\frac{1}{2}\beta(t)X_t$ shrinks the signal while the diffusion coefficient $\sqrt{\beta(t)}$ adds noise, keeping the total variance approximately constant throughout the process.
- **Terminal distribution:** Approximately $\mathcal{N}(0, I)$ at $t = T$.
- **Character:** The most "balanced" process — signal and noise trade off smoothly.

### 1.2 VE-SDE (Variance Exploding)

$$dX_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,dW_t$$

- **Origin:** Continuous-time limit of NCSN (Song & Ermon, 2019).
- **Behavior:** Zero drift — the data is left in place and noise is simply piled on top of it with an increasing schedule $\sigma(t)$. The variance of $X_t$ grows without bound.
- **Terminal distribution:** A very wide Gaussian (variance $\sigma^2(T) \gg 1$) centered near zero.
- **Character:** Conceptually simpler (no drift to implement), but the exploding variance can make score estimation harder at large $t$.

### 1.3 Sub-VP SDE

$$dX_t = -\tfrac{1}{2}\beta(t)\,X_t\,dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})}\,dW_t$$

- **Origin:** Introduced in Song et al. (ICLR 2021) alongside VP and VE.
- **Behavior:** Same drift as VP-SDE, but the diffusion coefficient is *strictly smaller* — it's modulated by a factor that starts at 0 and approaches the VP coefficient from below.
- **Terminal distribution:** The variance of $X_t$ is always bounded *below* the VP variance, hence "sub-VP."
- **Character:** Tighter concentration around the ODE trajectory; often yields better log-likelihoods. Theoretically interesting because it interpolates between the deterministic (ODE) and fully stochastic (VP) regimes.

### 1.4 CLD — Critically-Damped Langevin Diffusion

$$\begin{cases} dX_t = M^{-1}V_t\,dt \\ dV_t = -\left(\Gamma M^{-1} V_t + \beta(t) X_t\right)dt + \sqrt{2\Gamma}\,dW_t \end{cases}$$

- **Origin:** Dockhorn, Vahdat & Kreis (ICLR 2022).
- **Behavior:** The data $X_t$ is augmented with an auxiliary *momentum* variable $V_t$. Noise enters **only through $V_t$**; the data evolves deterministically conditioned on the momentum. The "critical damping" condition ($\Gamma = 2\sqrt{M\beta}$) produces the smoothest possible trajectories for $X_t$ — borrowed from classical control theory.
- **Score target:** Instead of $\nabla_x \log p_t(x)$, the network learns $\nabla_v \log p_t(v \mid x)$ — a fundamentally different object.
- **Reverse SDE:** Anderson's theorem must be applied to the *joint* $(X_t, V_t)$ system; the corresponding Fokker–Planck equation becomes a **Kramers equation**.
- **Character:** The richest theoretical variant. The second-order structure, Hamiltonian connection, and modified score target give substantial material for derivations.

### What Axis 1 demonstrates

By comparing these four on the same architecture and dataset, you directly show how the choice of $f(x,t)$ (drift) and $g(t)$ (diffusion coefficient) in the forward Itô SDE propagates through Anderson's reversal formula into the generative model. Each variant requires re-deriving the reverse SDE, the score-matching loss, and the probability-flow ODE — so the theory section of the monograph writes itself naturally from the experiments.

---

## Axis 2 — Reverse-Process Sampler (no retraining needed)

Once a model is trained (i.e., the score $s_\theta(x,t) \approx \nabla_x \log p_t(x)$ is learned), you generate samples by solving the *reverse-time* equation numerically. Different numerical methods produce different quality-vs-cost tradeoffs **from the exact same trained weights**.

Each forward SDE produces exactly **two** reverse-time equations, both uniquely determined by Anderson's theorem:

- The **reverse SDE** (stochastic): $dX_t = [f(X_t, t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{W}_t$
- The **probability-flow ODE** (deterministic): $dX_t = [f(X_t, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)]\,dt$

Both produce valid samples from the data distribution. The four samplers below are different numerical discretizations of these two equations.

### 2.1 Euler–Maruyama (EM)

The simplest SDE discretization:

$$X_{t-\Delta t} = X_t + \left[f(X_t, t) - g(t)^2\,s_\theta(X_t, t)\right]\Delta t + g(t)\sqrt{\Delta t}\,Z, \quad Z \sim \mathcal{N}(0, I)$$

- **Order:** Strong order 0.5, weak order 1.0.
- **Character:** One score evaluation per step, plus a noise injection. Simple but needs many steps ($\sim$1000) for high quality. This is the "pure stochastic" baseline.

### 2.2 Probability-Flow ODE (Euler)

Drop the noise term entirely:

$$X_{t-\Delta t} = X_t + \left[f(X_t, t) - \tfrac{1}{2}g(t)^2\,s_\theta(X_t, t)\right]\Delta t$$

- **Key insight:** Song et al. (2021) proved this deterministic ODE has the **same marginal distributions** $p_t(x)$ as the SDE at every time $t$. So it generates valid samples without any stochasticity in the reverse pass.
- **Character:** Deterministic — same initial noise always gives same output. One score evaluation per step. This is essentially what DDIM computes (Song, Meng & Ermon, 2021 showed DDIM is the first-order discretization of this ODE).

### 2.3 Heun's Method (2nd-order ODE)

A two-stage Runge–Kutta method applied to the probability-flow ODE:

$$\hat{X} = X_t + \Delta t \cdot d(X_t, t) \quad \text{(predictor)}$$
$$X_{t-\Delta t} = X_t + \tfrac{\Delta t}{2}\left[d(X_t, t) + d(\hat{X}, t-\Delta t)\right] \quad \text{(corrector)}$$

where $d(x,t)$ is the ODE drift.

- **Cost:** Two score evaluations per step (2 NFE).
- **Character:** The default sampler in EDM (Karras et al., 2022). Much better accuracy per step than Euler — typically matches Euler's quality in half the steps.

### 2.4 Predictor–Corrector (PC)

Combine an SDE step (predictor) with Langevin MCMC corrections:

1. **Predict:** Take one Euler–Maruyama step of the reverse SDE.
2. **Correct:** Run $K$ steps of Langevin dynamics at the current noise level:

$$X \leftarrow X + \epsilon\,s_\theta(X, t) + \sqrt{2\epsilon}\,Z$$

- **Origin:** Introduced in Song et al. (ICLR 2021), §4.
- **Character:** The Langevin corrections refine the sample toward the true $p_t(x)$ at each noise level, compensating for discretization error. More expensive ($1 + K$ NFE per step) but can achieve the highest quality.

### What Axis 2 demonstrates

From the same trained model, you compare:

| Sampler | Stochastic? | NFE / step | Theoretical order | Key tradeoff |
|---|---|---|---|---|
| Euler–Maruyama | Yes | 1 | Weak 1.0 | Simplest SDE baseline |
| Prob-Flow ODE (Euler) | No | 1 | 1.0 | Deterministic, same marginals |
| Heun (2nd-order ODE) | No | 2 | 2.0 | Best accuracy per NFE |
| Predictor–Corrector | Yes | 1 + K | Adaptive | Highest quality ceiling |

This axis is a direct study of **numerical methods for SDEs and ODEs** — convergence order, step-size sensitivity, the stochastic-vs-deterministic sampling tradeoff — all core stochastic calculus topics.

---

## The Combined Matrix

|  | EM (SDE) | PF-ODE Euler | Heun (2nd) | Pred.–Corr. |
|---|---|---|---|---|
| **VP-SDE** | ✓ | ✓ | ✓ | ✓ |
| **VE-SDE** | ✓ | ✓ | ✓ | ✓ |
| **sub-VP** | ✓ | ✓ | ✓ | ✓ |
| **CLD** | ✓ | ✓ | ✓ | ✓ |

16 cells total. Each cell is evaluated across both phases:

- **Phase 1 (2D):** visualize score fields, particle trajectories, and density reconstruction. For the Gaussian mixture, compare against the analytical true score.
- **Phase 2 (images):** measure FID and IS at matched NFE budgets (e.g., 50, 100, 250, 1000 function evaluations) and produce qualitative sample grids.

The poster gets the 2D trajectory visualizations and a FID heatmap; the monograph gets a theory chapter per forward process, a numerical-methods chapter for the samplers, and experimental chapters for both phases.

---

## Suggested Timeline (8 weeks)

| Week | Milestone |
|---|---|
| 1–2 | Build the MLP, training loop, and VP-SDE forward/reverse pipeline for 2D data. Get end-to-end generation working on Swiss roll. |
| 3–4 | Add VE-SDE, sub-VP, and CLD variants. Implement all four samplers. Run the full 4×4 matrix on all three 2D datasets. Produce score-field and trajectory visualizations. |
| 5 | Set up the U-Net (from library) and train VP-SDE on CIFAR-10 or FashionMNIST. Verify image generation works. |
| 6 | Train VE-SDE, sub-VP, and CLD on images. Run sampler ablation. Compute FID/IS. |
| 7–8 | Write monograph and design poster. Buffer for re-runs or debugging. |

---

## Key References per Variant

| Variant | Primary paper |
|---|---|
| VP-SDE / VE-SDE / sub-VP | Song et al., "Score-Based Generative Modeling through SDEs," ICLR 2021 |
| CLD | Dockhorn et al., "Score-Based Generative Modeling with CLD," ICLR 2022 |
| DDIM / PF-ODE | Song, Meng & Ermon, "Denoising Diffusion Implicit Models," ICLR 2021 |
| Heun sampler | Karras et al., "Elucidating the Design Space (EDM)," NeurIPS 2022 |
| Predictor–Corrector | Song et al. (ICLR 2021), §4 |
| Anderson's reversal | Anderson, "Reverse-time diffusion equation models," *Stoch. Proc. Appl.* 1982 |
| Original U-Net | Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015 |
| DDPM (U-Net adaptation) | Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020 |