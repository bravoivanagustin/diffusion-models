# Diffusion Models Meet Stochastic Calculus: A Literature Map for Image Generation

## TL;DR
- The single most-cited theoretical bridge from DDPM to stochastic calculus is **Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021, Outstanding Paper)** — it unifies NCSN and DDPM as discretizations of variance-exploding (VE) and variance-preserving (VP) SDEs, derives the reverse-time SDE from Anderson (1982), and introduces the probability-flow ODE; this is the paper any reviewer expects to see cited first.
- Practical SOTA image generation today rests on three "design-space" streams: Karras et al.'s **EDM (NeurIPS 2022) and EDM2 (CVPR 2024)** preconditioning/sampler analysis, the **flow-matching / rectified-flow / stochastic-interpolants** trio (Lipman 2023; Liu, Gong & Liu 2023; Albergo & Vanden-Eijnden 2023) used in Stable Diffusion 3, and **fast ODE solvers** (DDIM, DPM-Solver, DPM-Solver++, UniPC) plus **consistency models** for 1–4 step sampling.
- The most active 2024–2026 frontier extends the stochastic forward process beyond Brownian motion: **Lévy / α-stable noise** (Yoon et al. NeurIPS 2023; Shariatian et al. ICLR 2025), **fractional Brownian motion** (Nobis et al. NeurIPS 2024), **jump-diffusions** (Baule 2025), **Riemannian SDEs** (De Bortoli et al. NeurIPS 2022), **critically-damped Langevin** (Dockhorn et al. ICLR 2022), and **Schrödinger-bridge / bridge-matching** formulations (De Bortoli et al. NeurIPS 2021; Shi et al. NeurIPS 2023; Peluchetti JMLR 2023).

## Key Findings

The literature naturally splits into three layers. **Layer 1** (foundational SDE/ODE theory) is small, well-defined, and dominated by Song-and-collaborators plus the 1982 Anderson result that makes the reverse-time SDE possible at all. **Layer 2** (design-space and sampler engineering) is where most practical FID gains have come from since 2022 — EDM's preconditioning, DPM-Solver/DPM-Solver++/UniPC for ODE solvers, DDIM as a first-order ODE solver, flow matching/rectified flow as a re-parameterization of the underlying ODE, and consistency models for single-step sampling. **Layer 3** (niche stochastic processes) is the live research frontier: heavy-tailed Lévy / α-stable processes, fractional Brownian motion (with Hurst parameter H ≠ ½), Poisson-jump processes, Riemannian Brownian motion, and bridge / Schrödinger-bridge constructions that replace the Ornstein–Uhlenbeck forward SDE with a finite-time entropic-OT process.

A reviewer-grade citation list should include essentially every paper below; the **Details** section is organized chronologically within each theme so the reader can see how ideas built up.

## Details

### 1. Theoretical SDE/ODE formulations

**1.1 Anderson (1982) — the mathematical prerequisite.** Brian D. O. Anderson, "Reverse-time diffusion equation models," *Stochastic Processes and their Applications* 12(3):313–326 (1982). Proves that for a forward Itô SDE dX_t = f(X_t,t)dt + g(t)dW_t, the time-reversed process is itself an Itô SDE with drift f(x,t) − g(t)²∇log p_t(x) and the same diffusion coefficient. Every modern score-based / SDE diffusion paper invokes this result; Song et al. (2021) credit Anderson explicitly.

**1.2 Song & Ermon (2019, NeurIPS) — NCSN.** Yang Song and Stefano Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," *NeurIPS 2019* (Oral). Introduces Noise Conditional Score Networks trained with denoising score matching across multiple Gaussian noise scales, sampled via annealed Langevin dynamics. This is the discrete-time precursor to the variance-exploding SDE.

**1.3 Song & Ermon (2020, NeurIPS) — Improved NCSN.** "Improved Techniques for Training Score-Based Generative Models," NeurIPS 2020. Provides the noise-schedule and EMA practices that stabilized score networks and enabled the leap to high-resolution images.

**1.4 Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (ICLR 2021) — the unified SDE framework.** *arXiv:2011.13456*, ICLR 2021 Outstanding Paper. The single most important paper after DDPM. It formalizes:
- **VE-SDE** (variance-exploding, generalizing NCSN): dX = √(d[σ²(t)]/dt) dW.
- **VP-SDE** (variance-preserving, generalizing DDPM): dX = −½β(t)X dt + √β(t) dW.
- **sub-VP SDE**: a strictly tighter-variance variant introduced in the same paper that often yields better likelihoods.
- **Reverse-time SDE** for sampling (via Anderson 1982).
- **Probability-flow ODE**: a deterministic ODE with the *same* marginal distributions p_t(x) as the SDE; enables exact likelihood via the instantaneous change-of-variables formula and connects diffusion models to neural ODEs / continuous normalizing flows.
- **Predictor-corrector samplers** combining numerical SDE solvers with Langevin MCMC corrections.

State-of-the-art at publication: FID 2.20 on CIFAR-10. This paper is the conceptual hub everything else in Layer 1/2 connects to.

**1.5 Song, Durkan, Murray & Ermon (NeurIPS 2021) — Maximum-Likelihood Training of Score Models.** Spotlight paper showing how to weight the score-matching loss to obtain a proper upper bound on negative log-likelihood, tying score-based models to likelihood-based generative modeling.

**1.6 De Bortoli, Thornton, Heng & Doucet (NeurIPS 2021) — Diffusion Schrödinger Bridge (DSB).** *arXiv:2106.01357*. Replaces the long-time-horizon OU SDE with a finite-time **Schrödinger bridge** — the entropy-regularized optimal-transport problem on path space — and solves it via an Iterative Proportional Fitting / score-matching scheme. Removes the assumption that the forward SDE has converged to a Gaussian at time T.

**1.7 Theoretical convergence guarantees.** A cluster of 2022–2024 theory papers established that diffusion models can sample efficiently from essentially any data distribution given accurate scores:
- **Chen, Chewi, Li, Li, Salim & Zhang (ICLR 2023), "Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions"** (*arXiv:2209.11215*) — proves polynomial-time TV-distance convergence assuming only L²-accurate scores and finite second moments.
- **Lee, Lu & Tan (ALT 2023), "Convergence of Score-Based Generative Modeling for General Data Distributions."**
- **Chen, Lee & Lu (ICML 2023), "Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions."**
- **Benton, De Bortoli, Doucet & Deligiannidis (2023)** — convergence of probability-flow ODE samplers.

These results justify why diffusion models work in practice and quantify how score-approximation error, discretization error, and forward-mixing error compose.

**1.8 Connections to optimal transport.** Beyond Schrödinger bridges, **Khrulkov & Babenko (ICLR 2023) "Understanding DDPMs with Optimal Transport"** and **Lavenant & Santambrogio (2022)** analyze the probability-flow ODE through an OT lens; **Kwon, Fan & Lee (NeurIPS 2022) "Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance"** shows the implicit OT minimization happening inside SGMs.

### 2. Practical architectural and stochastic-module variations

**2.1 Ho, Jain & Abbeel (NeurIPS 2020) — DDPM.** (Already read.) Listed only to anchor the lineage; corresponds to discretizing the VP-SDE with a specific β-schedule.

**2.2 Song, Meng & Ermon (ICLR 2021) — DDIM.** *arXiv:2010.02502*. Constructs a non-Markovian forward process whose ELBO objective is identical to DDPM's, but whose reverse process can be made *deterministic*. As the DDIM authors state in the abstract, the method can "produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs." Salimans & Ho (2022) and Lu et al. (2022) later proved DDIM is the **first-order discretization of the probability-flow ODE** — making DDIM a numerical solver, not a separate model.

**2.3 Dhariwal & Nichol (NeurIPS 2021) — "Diffusion Models Beat GANs on Image Synthesis."** Introduces classifier guidance and ablation of architecture/noise-schedule choices.

**2.4 Nichol & Dhariwal (ICML 2021) — Improved DDPM.** Learns the reverse-process variance and introduces the cosine noise schedule still used widely today.

**2.5 Dockhorn, Vahdat & Kreis (ICLR 2022 Spotlight, top 0.4% by reviewer score) — Critically-Damped Langevin Diffusion (CLD).** *arXiv:2112.07068*. Augments the data x_t with an auxiliary momentum/velocity v_t and runs a *Hamiltonian-like* underdamped Langevin SDE in the joint (x,v) space, with noise injected only into v_t. The "critical damping" choice — borrowed from classical control theory — produces the smoothest possible trajectories for x_t, making the denoising score-matching target simpler (one only needs ∇_v log p_t(v|x)). Per the NVIDIA Research project page, "Our CLD-based SGMs achieve FID scores of 2.25 and 2.23 using probability flow ODE sampling and generative SDE sampling, respectively" on CIFAR-10, outperforming all published diffusion results at comparable NFE budgets.

**2.6 Karras, Aittala, Aila & Laine (NeurIPS 2022) — EDM.** *arXiv:2206.00364*, "Elucidating the Design Space of Diffusion-Based Generative Models." Disentangles the design space into (a) the noise schedule σ(t), (b) the network preconditioning (input scaling, output skip connection, loss weighting), (c) the time-step sampling distribution at training, and (d) the ODE/SDE sampler. Recommends Heun's 2nd-order ODE solver, σ(t)=t, and a specific log-normal training-time σ-distribution. New SOTA FID **1.79 (class-cond) and 1.97 (uncond) on CIFAR-10** with 35 NFE/image. EDM's parameterization is the de facto standard for diffusion training in 2024–2026.

**2.7 Karras, Aittala, Lehtinen, Hellsten, Aila & Laine (CVPR 2024) — EDM2.** "Analyzing and Improving the Training Dynamics of Diffusion Models," *arXiv:2312.02696*. Hypersphere-constrained network parameterization preserving expected activation/weight/update magnitudes plus post-hoc EMA reconstruction. Improves the previous record FID on ImageNet-512 — held by the original EDM (Karras et al., NeurIPS 2022), Table 1, at 2.41 — to **1.81** with fast deterministic sampling.

**2.8 Rombach, Blattmann, Lorenz, Esser & Ommer (CVPR 2022) — Latent Diffusion / Stable Diffusion.** *arXiv:2112.10752*. Architectural rather than stochastic-calculus contribution: train a VQ/KL autoencoder, then run the diffusion SDE in a lower-dimensional latent space with cross-attention conditioning. Unlocks high-resolution text-to-image; the open-source release as Stable Diffusion is the practical reason most subsequent papers use LDM backbones.

**2.9 Lu, Zhou, Bao, Chen, Li & Zhu (NeurIPS 2022) — DPM-Solver.** *arXiv:2206.00927*. Reformulates the probability-flow ODE so its linear (drift) term is integrated analytically (exponential integrator), leaving only the neural-network term to a numerical solver. Provides a dedicated high-order solver with convergence guarantees: FID **4.70 in 10 NFE / 2.87 in 20 NFE on CIFAR-10**, a 4–16× speedup vs. previous training-free samplers.

**2.10 Lu, Zhou, Bao, Chen, Li & Zhu (2022) — DPM-Solver++.** *arXiv:2211.01095*. Multistep variant adapted to **guided** sampling at large CFG scales (where previous high-order solvers became unstable); employs data-prediction parameterization and thresholding. Default sampler in Stable Diffusion / many open-source text-to-image stacks. A journal version appeared in *Machine Intelligence Research* (2025).

**2.11 Zhao, Bai, Rao, Zhou & Lu (NeurIPS 2023) — UniPC.** *arXiv:2302.04867*. A predictor-corrector ODE framework: the unified corrector UniC can be applied after any existing DPM sampler to gain an extra order of accuracy with no additional NFE; UniP is the arbitrary-order predictor. **FID 3.87 on CIFAR-10 (uncond) and 7.51 on ImageNet-256 (cond) at 10 NFE.**

**2.12 Song, Dhariwal, Chen & Sutskever (ICML 2023) — Consistency Models.** *arXiv:2303.01469*. Trains a network f(x_t, t) constrained to satisfy a *self-consistency* condition along entire probability-flow ODE trajectories, so f(x_t, t) ≈ x_0 for any t. Enables **one-step** or few-step generation while remaining a principled object from the SDE framework. Trained either by distilling a pretrained diffusion model (consistency distillation) or from scratch (consistency training).

**2.13 Song & Dhariwal (ICLR 2024 Oral, Top 1.2%) — Improved Consistency Training (iCT).** *arXiv:2310.14189*. Fixes a theoretical flaw in CT, removes the EMA teacher, replaces LPIPS with Pseudo-Huber loss, introduces a **lognormal noise schedule** on diffusion timesteps, and progressively doubles discretization. Pushes one-step FID to **2.51 on CIFAR-10 and 3.25 on ImageNet 64×64**.

**2.14 Lu & Song (ICLR 2025, arXiv:2410.11081) — sCM (Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models).** Introduces the **TrigFlow** parameterization unifying EDM and Flow-Matching SDE/ODE formulations; eliminates discretization error inherent in discrete CMs; identifies and fixes continuous-time training instabilities. Scales CMs to 1.5B parameters; two-step sampling achieves **FID 2.06 (CIFAR-10), 1.48 (ImageNet 64), 1.88 (ImageNet 512)** — closing most of the gap to multi-step diffusion.

**2.15 Lipman, Chen, Ben-Hamu, Nickel & Le (ICLR 2023) — Flow Matching.** *arXiv:2210.02747*. A simulation-free objective for training continuous normalizing flows by regressing the *conditional* vector field of fixed probability paths between data and noise. The marginal vector field equals an expectation of the conditional one (a clean theoretical result), so gradients match without sampling trajectories. Importantly, diffusion paths are a *special case* of FM; OT-displacement-interpolant paths are straighter, train faster, and integrate with fewer steps. Foundational for SD3.

**2.16 Liu, Gong & Liu (ICLR 2023 Spotlight) — Rectified Flow.** *arXiv:2209.03003*. Concurrent with FM. Learns an ODE whose trajectories are *straight lines* connecting noise–data pairs (z_t = (1−t)x_0 + tx_1), and introduces **reflow**, an iterative procedure that re-pairs noise/data using a learned flow to straighten trajectories further — eventually enabling 1-step generation. Distilled 2-Rectified Flow attains FID 4.85 on CIFAR-10 in a single Euler step.

**2.17 Albergo & Vanden-Eijnden (ICLR 2023) — Stochastic Interpolants (InterFlow).** *arXiv:2209.15571*, "Building Normalizing Flows with Stochastic Interpolants." Concurrent with FM/RF. Introduces stochastic interpolants x_t = I_t(x_0, x_1) + γ(t)z and derives a quadratic-loss objective for both the velocity field and the score, without ODE backpropagation.

**2.18 Albergo, Boffi & Vanden-Eijnden (JMLR 2025) — Stochastic Interpolants: A Unifying Framework.** *arXiv:2303.08797*. Extends interpolants to bridge *any* two prescribed densities in finite time. The interpolant's time-dependent density satisfies a first-order transport equation **and** a family of forward/backward Fokker–Planck equations with *tunable* diffusion coefficient — giving deterministic ODE or stochastic SDE generative models with adjustable noise level from a single training run. This is now widely viewed as the most general formulation that contains diffusion, flow matching, and rectified flow as special cases.

**2.19 Esser, Kulal, Blattmann, Entezari, Müller, Saini, Levi, Lorenz, Sauer, Boesel, Podell, Dockhorn, English, Lacey, Goodwin, Marek & Rombach (ICML 2024) — Stable Diffusion 3 / Scaling Rectified Flow Transformers.** *arXiv:2403.03206*. Introduces the **MM-DiT** transformer with separate text/image streams and improves rectified-flow training by biasing timestep sampling toward perceptually difficult mid-range timesteps (t ≈ 0.5) using **logit-normal** and **mode-based** densities. Per Table 5 of the paper, "Our largest model (depth=38) outperforms all current open models and DALLE-3 (Betker et al., 2023) on GenEval," confirming the 8B MM-DiT achieves the top overall GenEval score among compared systems including SDXL, DALL·E 3, and DeepFloyd IF. The rectified-flow / flow-matching family is now the dominant training paradigm for frontier image models.

### 3. Niche, "exotic" and cutting-edge stochastic formulations

**3.1 De Bortoli, Mathieu, Hutchinson, Thornton, Teh & Doucet (NeurIPS 2022) — Riemannian Score-Based Generative Modelling (RSGM).** *arXiv:2202.02763*. Extends SGMs to compact Riemannian manifolds: forward noising is Brownian motion *on the manifold*, the reverse-time formula is generalized via the manifold's Laplace–Beltrami operator and Levi-Civita connection, and likelihood is computed via neural ODEs on manifolds. Applied to earth/climate spherical data and protein conformations.

**3.2 Thornton, Hutchinson, Mathieu, De Bortoli, Teh & Doucet (2022) — Riemannian Diffusion Schrödinger Bridge.** *arXiv:2207.03024*. Combines DSB with the manifold formulation.

**3.3 Yoon, Park, Kim & Lim (NeurIPS 2023 Spotlight) — Lévy-Itō Model (LIM).** "Score-based Generative Models with Lévy Processes," NeurIPS 2023. Replaces Brownian motion with an isotropic **α-stable Lévy process** (a heavy-tailed pure-jump or jump-diffusion process). Derives the exact reverse-time SDE driven by a Lévy process and a **fractional denoising score-matching** loss. Reports FID 1.58 on CelebA (DDPM baseline: 3.21) and dramatically better recall on class-imbalanced data (CIFAR-10-LT). The heavy tails enable larger exploratory moves, helping with mode coverage.

**3.4 Shariatian, Simsekli & De Bortoli (ICLR 2025) — Heavy-Tailed Diffusion with Denoising Lévy Probabilistic Models (DLPM).** *arXiv:2407.18609*. A discrete-time, more accessible α-stable diffusion model that achieves better tail coverage than LIM with simpler mathematics. Released alongside the DLPM code as a counterpart to LIM's continuous-time SDE.

**3.5 Baule (2025) — Generative Modelling with Jump-Diffusions.** *arXiv:2503.06558*, single-author preprint. Generalizes score-based diffusion to **finite-activity Lévy processes** = Gaussian diffusion + super-imposed Poisson jumps. Derives a **generalized score function** that depends on the jump-amplitude distribution and can be learned with a plain MSE loss; provides both probability-flow ODE and SDE forms. Demonstrates closed-form implementation for Laplace-distributed jumps (the "JL" model). Not yet peer-reviewed as of May 2026.

**3.6 Nobis, Springenberg, Aversa, Detzel, Daems, Murray-Smith, Nakajima, Lapuschkin, Ermon, Birdal, Opper, Knochenhauer, Oala & Samek (NeurIPS 2024) — Generative Fractional Diffusion Models (GFDM).** *arXiv:2310.17638*. First continuous-time score model driven by an approximation of **fractional Brownian motion (fBM)** with Hurst index H ∈ (0,1) (H = ½ recovers Brownian; H > ½ gives long-range dependence and super-diffusive behavior; H < ½ gives roughness). Uses a Markov approximation MA-fBM representing fBM as a stochastic integral of correlated Ornstein–Uhlenbeck processes, enabling a tractable reverse-time model and continuous re-parameterization. Improves pixel-wise diversity and FID on imbalanced datasets.

**3.7 Bansal et al. (NeurIPS 2023) — Cold Diffusion.** *arXiv:2208.09392*. Replaces additive Gaussian noise with *deterministic* degradations (Gaussian blur, masking, downsampling, animorphosis, snow). Generative models still work, *calling into question whether stochasticity is essential* — and producing a generalized framework that inverts arbitrary degradation processes. From a stochastic-calculus perspective, this is the limiting case g(t) → 0 of the SDE.

**3.8 Xu, Liu, Tegmark & Jaakkola (NeurIPS 2022) — Poisson Flow Generative Models (PFGM).** *arXiv:2209.11178*. Replaces the diffusion SDE with a deterministic flow along electric-field lines of a **Poisson equation** in a space augmented by one extra dimension. State-of-the-art among normalizing flows on CIFAR-10 (IS 9.68, FID 2.35) with 10–20× faster sampling than SDE approaches. **PFGM++ (Xu et al. ICML 2023)** unifies PFGM and diffusion as a one-parameter family indexed by an augmented-dimension D.

**3.9 Shi, De Bortoli, Campbell & Doucet (NeurIPS 2023) — Diffusion Schrödinger Bridge Matching (DSBM).** *arXiv:2303.16852*. Introduces **Iterative Markovian Fitting (IMF)** for the dynamic Schrödinger Bridge problem and DSBM as its numerical realization. IMF alternates Markovian and reciprocal projections; recovers DDM, flow matching, and bridge matching as limiting cases. With Peluchetti's IDBM, this work founded the "bridge matching" sub-field.

**3.10 Peluchetti (JMLR 2023) — Iterated Diffusion Bridge Mixtures.** "Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling," *JMLR 24(374):1–51*. Single-author. Sampling-based iterative algorithm for the dynamic Schrödinger Bridge problem; each iterate is already a valid transport between marginals. With Shi et al. 2023, the joint foundation of bridge matching.

**3.11 Liu, Wu, Ye & Liu (NeurIPS 2023) — I²SB: Image-to-Image Schrödinger Bridge.** Direct application of SB to paired image-translation problems, generalizing diffusion models for image-to-image translation, where the SB endpoints are degraded and clean images.

**3.12 Hoogeboom & Salimans (ICLR 2023) — Blurring Diffusion Models.** *arXiv:2209.05557*. Replaces additive Gaussian noise with anisotropic *heat-equation* diffusion in frequency space; can be cast as an SDE with frequency-dependent drift. Improves perceptual quality for natural images by destroying high frequencies first.

**3.13 Daras et al. (TMLR 2023) — Soft Diffusion.** A unified framework for arbitrary linear corruption operators paired with additive Gaussian noise, generalizing both DDPM and Cold Diffusion.

**3.14 Other notable 2023–2025 entries that a thorough survey should cite:**
- **Pidstrigach (NeurIPS 2022) — Score-based generative models detect manifolds.** Theoretical analysis of when SGMs sample from low-dimensional manifolds.
- **Berner, Richter, Ullrich (2024) — "An Optimal Control Perspective on Diffusion-Based Generative Modeling,"** TMLR. Reformulates score-based modeling as a stochastic optimal control problem (Hamilton–Jacobi–Bellman perspective).
- **Holderrieth, Albergo & Jaakkola (2024) — Generator Matching.** Generalizes flow matching to arbitrary Markov generators (including jumps and on manifolds).
- **Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman & Chen (ICML 2023) — Multisample Flow Matching.** Uses mini-batch OT couplings to define straighter probability paths.
- **Tong, Malkin et al. (TMLR 2024) — Improving and generalizing flow-based generative models with minibatch optimal transport (CFM with OT).** Bridges between SI/FM and Schrödinger bridges via mini-batch OT.

## Recommendations

For a literature review or related-work section in a paper at this intersection, the **minimum citation set** any reviewer expects (besides Sohl-Dickstein 2015 and Ho 2020, which the user has already read) is:

1. **Foundational SDE/ODE**: Anderson (1982); Song & Ermon (NeurIPS 2019, NCSN); Song et al. (ICLR 2021, SDE unification); Song, Meng & Ermon (ICLR 2021, DDIM); Karras et al. (NeurIPS 2022, EDM).
2. **Samplers/solvers**: Lu et al. (NeurIPS 2022, DPM-Solver); Lu et al. (2022, DPM-Solver++); Zhao et al. (NeurIPS 2023, UniPC); Song et al. (ICML 2023, Consistency Models).
3. **Flow/interpolant family**: Lipman et al. (ICLR 2023, Flow Matching); Liu, Gong & Liu (ICLR 2023, Rectified Flow); Albergo & Vanden-Eijnden (ICLR 2023, Stochastic Interpolants); Esser et al. (ICML 2024, SD3).
4. **Theory**: Chen, Chewi et al. (ICLR 2023, convergence guarantees); De Bortoli et al. (NeurIPS 2021, DSB).
5. **One niche choice from**: CLD (Dockhorn et al. ICLR 2022), RSGM (De Bortoli et al. NeurIPS 2022), LIM (Yoon et al. NeurIPS 2023), GFDM (Nobis et al. NeurIPS 2024), PFGM++ (Xu et al. ICML 2023), or Bridge Matching (Shi et al. NeurIPS 2023; Peluchetti JMLR 2023), depending on which exotic stochastic process is most relevant to the writer's project.

**Staged reading plan for someone who has read DDPM and Sohl-Dickstein**:
- *Week 1 — get fluent in the SDE picture*: Anderson (1982) skim; Song et al. (ICLR 2021) read in depth; Karras et al. EDM (NeurIPS 2022) for design space.
- *Week 2 — samplers and DDIM as ODE*: DDIM (ICLR 2021); DPM-Solver / DPM-Solver++; UniPC; Consistency Models.
- *Week 3 — flow and interpolants*: Flow Matching; Rectified Flow; Stochastic Interpolants (the 2023 unifying paper); Esser et al. SD3.
- *Week 4 — niche*: CLD; RSGM; LIM/DLPM; GFDM; DSB + Bridge Matching.

**Benchmarks that should change the plan**: if the project is specifically about heavy-tailed data, *prioritize* LIM/DLPM and GFDM. If it is about geometric data (proteins, climate, 3D), *prioritize* RSGM and Riemannian Flow Matching. If it is about fast inference, *prioritize* consistency models (especially sCM) and DPM-Solver++. If it is about training stability at large scale, *prioritize* EDM2 and sCM.

## Caveats

- **Concurrent independent discoveries**: Flow Matching (Lipman), Rectified Flow (Liu et al.), and Stochastic Interpolants (Albergo & Vanden-Eijnden) appeared *within weeks of one another* (October 2022 arXiv preprints) and were published simultaneously at ICLR 2023. They are mathematically near-equivalent in their core ideas and should be co-cited.
- **Naming inconsistency**: "Score-based model," "diffusion model," and "diffusion probabilistic model" are used interchangeably in the literature for what is essentially the same SDE family. NCSN, DDPM, and Song et al. 2021 differ chiefly in noise scaling (VE vs. VP) and parameterization, not in fundamental mechanism.
- **Some 2024–2026 entries are preprints**: Baule (2025) jump-diffusions and Lu & Song (sCM, 2024) are arXiv preprints at the time of writing (sCM is accepted to ICLR 2025; Baule has no confirmed venue as of May 2026). Treat citations accordingly.
- **Author lists drift between arXiv versions and proceedings**: Notably, Nobis et al. (GFDM) added 3 authors between arXiv v1 (11 authors) and the NeurIPS 2024 final version (14 authors); use the 14-author list for formal citation. Esser et al. (SD3) similarly varies; the arXiv 17-author list and the ICML 14-author list both circulate.
- **Anderson's reverse-time SDE has subtler general forms**: For position- and time-dependent diffusion coefficients g(x,t), an additional term appears in the reverse drift (Haussmann & Pardoux, *Annals of Probability* 14(4):1188–1205, 1986). Most diffusion papers use g(t) only, where Anderson's stated result is exact.
- **Theoretical convergence results assume L²-accurate scores**: Real-world score networks may not satisfy this, so polynomial-time guarantees are existence results, not direct quality predictions.