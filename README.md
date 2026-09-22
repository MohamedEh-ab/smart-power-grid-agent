# EGrid-118: Smart Power Grid Controller Agent and Energy Dispatcher

**Augmented Lagrangian OPF-RL with Graph Attention Networks & Proximal Policy Optimization**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22885476.svg)](https://doi.org/10.5281/zenodo.22885476)

## What the System Does

EGrid-118 is a deep reinforcement learning controller that manages real-time power dispatch across a 118-
bus electricity network. The agent continuously monitors every bus's voltage, line thermal loading, and generator
states, then issues three types of actions every control step: active power redispatch (ΔP), voltage setpoint
adjustment (V), and renewable curtailment fraction. It learns entirely through interaction, receiving reward signals
that encode both engineering objectives (keep voltage within ±6%, avoid line overloads, minimize cost) and equity
constraints (fair load distribution across buses).

The system evolves through four phases of increasing sophistication, beginning with a simple reward-based
baseline and culminating in a fully augmented Lagrangian formulation with spatially-aware graph neural networks

## Table of contents

1. [How to use this project (library + notebook + environment)](#1-how-to-use-this-project-library--notebook--environment)
2. [What the notebook covers](#2-what-the-notebook-covers)
3. [Project overview](#3-project-overview)
4. [Mathematical logic and model architecture](#4-mathematical-logic-and-model-architecture)
5. [Curriculum training: the episode waves](#5-curriculum-training-the-episode-waves)
6. [Inference and evaluation](#6-inference-and-evaluation)
7. [Repository structure](#7-repository-structure)
8. [Notes](#8-notes)

---

## 1) How to use this project (library + notebook + environment)

This repository has two main entry points:

- **`smart_grid_gatpo_lib.py`**: a reusable Python library for training/inference with a GAT + PPO controller for smart grid optimal power flow.
- **`Smart_Grid_Agent.ipynb`**: a research-style notebook that explains the full method, trains/evaluates the agent, and generates analysis/visualizations.

### For a normal audience

- Think of this as an **AI controller for electricity grids**.
- The notebook shows how the AI learns to keep the grid stable and fair under difficult conditions (like supply shocks).
- The library is the "engine" you can reuse in your own scripts/projects.

### For experienced AI / power-system engineers

- The project implements **GAT-PPO** for constrained OPF-style dispatch on **`l2rpn_wcci_2022`**.
- It uses a **primal-dual + augmented Lagrangian** reward design
  (automatic constraint handling during RL training), with curriculum training
  and spatial clustering.
- The library exposes modular components for training and deployment:
  `YBusBuilder`, `BusClusterer`, `Grid2OpAdapter`, `PandapowerAdapter`, and
  `GATPO` (the repository API class name for the GAT-PPO controller).

### Environment and backend used

- **Python version**: Python 3.10+ is recommended (Google Colab default runtime is suitable).
- **Primary development/runtime environment**: Google Colab (CPU/GPU runtime), as documented in the notebook.
- **Grid simulation environment**: **Grid2Op** (`l2rpn_wcci_2022`). The dataset is downloaded automatically by Grid2Op on first use, so the first run needs internet access.
- **Power-flow backend**: **LightSim2Grid**.

### Installing dependencies

The repository does not ship a `requirements.txt`. Pick one of the two options below.

#### Option A: pip (Google Colab or a plain virtual environment)

Core dependencies used directly in the notebook/library:

```bash
pip install lightsim2grid grid2op networkx seaborn numpy pandas torch gymnasium matplotlib plotly scikit-learn
```

Optional (for the model-graph visualization in the notebook):

```bash
pip install torchviz
```

#### Option B: conda (recommended for local machines)

Conda lets you create an isolated, reproducible environment without a `requirements.txt` file.

**B1. Quick setup (three commands).** Install the scientific stack from conda first, then the two Grid2Op packages with pip (conda packages first, pip last, is the safest order):

```bash
conda create -n smartgrid python=3.10 -y
conda activate smartgrid

conda install -c conda-forge -y numpy pandas scikit-learn matplotlib seaborn plotly networkx gymnasium pytorch jupyterlab ipykernel
pip install lightsim2grid grid2op
```

For a GPU build of PyTorch, use the selector on <https://pytorch.org> to get the command matching your CUDA version, and run it in place of the `pytorch` package above.

**B2. Reproducible setup from an `environment.yml` file.** Save the following as `environment.yml` in the repository root:

```yaml
name: smartgrid
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - networkx
  - gymnasium
  - pytorch
  - jupyterlab
  - ipykernel
  - pip
  - pip:
      - lightsim2grid
      - grid2op
      # - torchviz   # optional, see below
```

Then create and activate the environment:

```bash
conda env create -f environment.yml
conda activate smartgrid
python -m ipykernel install --user --name smartgrid --display-name "Python (smartgrid)"
```

To update it later after editing the file: `conda env update -f environment.yml --prune`.

**B3. Verify the installation.**

```bash
python -c "import torch, grid2op, lightsim2grid, gymnasium, sklearn; print('torch', torch.__version__, '| grid2op', grid2op.__version__)"
```

**B4. Pin exact versions for reproducibility.** Once everything works on your machine, export the environment so others (and future you) can recreate it exactly:

```bash
# Portable: only the packages you explicitly installed (works across OSes)
conda env export --from-history > environment.yml

# Exact: every package and build string (same OS/architecture only)
conda env export > environment.lock.yml
```

**Optional: `torchviz`** (model graph visualization) needs the Graphviz system binaries as well:

```bash
conda install -c conda-forge -y graphviz python-graphviz
pip install torchviz
```

> The first cells of the notebook run `pip install` for a few packages. In a conda environment prepared as above they are already installed, so those cells are harmless and can be skipped.

### Basic usage

1. Install the requirements (Option A or B above).
2. Create the folders the notebook writes to and make sure the plant data file is in place:

   ```bash
   mkdir -p checkpoints output/output_csv env_needed_csv
   # env_needed_csv/egypt_power_plants_processed.csv must exist
   ```

3. Open `Smart_Grid_Agent.ipynb` in Colab/Jupyter to reproduce training and analysis. If you are training from scratch, comment out the `PPOGAT.load_checkpoint('./checkpoints/ckpt_alpha.pth')` line in the "Instantiate, Train, Evaluate" section, because that file only exists if you already have a checkpoint.
4. Import `smart_grid_gatpo_lib.py` in scripts to run inference or training loops with Grid2Op or pandapower adapters.

---

## 2) What the notebook covers

The notebook walks through:

1. Environment setup and package installation,
2. Grid environment/wrapper design,
3. GAT policy/value architecture,
4. PPO + (augmented) Lagrangian training loop,
5. Curriculum over different shock regimes,
6. Evaluation, baselines, ablations, and visual analytics.

---

## 3) Project overview

The project targets **real-time grid control** under operational constraints:

- Maintain voltage/thermal safety,
- Reduce blackout risk,
- Balance generation dispatch and renewable curtailment,
- Preserve fairness (load equity) under stress conditions.

It blends physics-informed graph learning with reinforcement learning to handle large, structured power-grid state/action spaces.

---

## 4) Mathematical logic and model architecture

**In plain words:** at every simulation step the agent looks at the grid (voltages, power flows, which lines are up, how much each generator is producing), decides how much every generator should produce, at what voltage, and how much renewable energy to curtail, and then the physics simulator computes what really happens. The agent is rewarded for keeping the lights on cheaply and fairly, and it is penalized when it pushes the grid outside safe limits. The penalty weights are not hand-tuned: they are adjusted automatically during training (the "Lagrange multipliers").

### 4.1 System at a glance

```
Grid2Op observation
   ├─ bus features   X_bus  [118 × 6]
   ├─ generator feats X_gen  [4 × 62 = 248]
   └─ Y_bus → normalized (G, B) edge tensor A  [2 × 118 × 118]     (rebuilt every step)
                     │
        Physics-informed GAT actor–critic  (271,270 trainable parameters)
                     │
   Gaussian policy over 186 actions in [0, 1]:
        [ P setpoints (62) | V setpoints (62) | curtailment (62) ]
                     │
        Safety layer (clip to generator limits, measures g_G)
                     │
        Grid2Op redispatch / prod_v / curtail  →  LightSim2Grid AC power flow
                     │
        next observation, reward, constraint signals (g_V, g_T, g_G, g_C)
```

### 4.2 Constrained decision problem

Dispatch is modeled as a constrained Markov decision process. The agent maximizes discounted task reward subject to four constraint signals $g_i \le 0$, $i \in \{V, T, G, C\}$ (voltage, thermal, generator limits, curtailment):

$$
\max_{\pi}\; J(\pi)=\mathbb{E}_{\pi}\Big[\sum_{t}\gamma^{t}\, r^{\text{task}}_t\Big]
\quad\text{s.t.}\quad \mathbb{E}_{\pi}\big[g_i(s_t,a_t)\big]\le 0,\;\; i\in\{V,T,G,C\}
$$

It is solved with a **primal-dual augmented Lagrangian** scheme:

$$
\mathcal{L}_{\rho}(\pi,\lambda)=J(\pi)-\sum_{i}\lambda_i\, g_i-\sum_{i}\frac{\rho_i}{2}\, g_i^{2}
$$

- **Primal step:** PPO improves the policy $\pi_\theta$ on the shaped reward (Section 4.8).
- **Dual step:** the multipliers $\lambda_i$ grow when a constraint is violated and decay when it is satisfied (Section 4.9).
- **Quadratic term:** $\frac{\rho_i}{2} g_i^2$ with $\rho_i = 5$ damps the "seesaw" oscillation between violating and over-correcting.

### 4.3 State and action

| Item | Definition |
|------|-----------|
| Bus features (per bus, 6) | $P/400$, $V-1$, $\theta/180°$, connectivity ratio, tripped-line fraction, load-bus flag. Gaussian noise ($\sigma=0.02$) is added to the first three (SCADA-style measurement error). |
| Generator features (per generator, 4) | $P/P_{max}$, $V/1.04$, renewable flag, headroom $=1-P/P_{max}$. Flattened to 248 values. |
| Edge tensor | Normalized $(\tilde G,\tilde B)$ from the current $Y_{bus}$ (Section 4.5). |
| Action $a\in[0,1]^{186}$ | $[a_P\ \Vert\ a_V\ \Vert\ a_C]$ (concatenation), each part of length 62. |

The action is decoded by the environment as follows:

- **Active power:** $P^{tgt}_k=\text{clip}(a_{P,k}\cdot P^{real}_{max,k},\,P^{grid}_{min,k},\,P^{grid}_{max,k})$. Non-renewable generators receive the redispatch $P^{tgt}_k-P^{dispatch}_k$.
- **Voltage setpoint:** $V_k = 0.90 + 0.20\, a_{V,k}$ p.u. (range 0.90 to 1.10).
- **Curtailment:** $a_{C,k}$ is the curtailment fraction, applied to renewable generators (solar, wind, hydro).
- **Suez supply shock:** before decoding, $a_P$ of every gas/oil generator is multiplied by $(1-\text{shock})$.
- **Random trip:** with probability `failure_prob` per step (0.05 in training), one random generator's $a_P$ is set to 0.

Generator capacities come from the EEHC 2022 plant table (`egypt_power_plants_processed.csv`), matched to the 62 WCCI generators by descending capacity, and scaled to the grid with $\text{cap\_scale}=\sum P^{Egypt}_{max}/\sum P^{WCCI}_{max}$.

### 4.4 Constraint functions

All four are computed after the power flow of each step, using the new observation.

| Constraint | Definition (as implemented) | Meaning |
|-----------|-----------------------------|---------|
| $g_V$ | $0.4\,\overline{(v_i-1)^2}+0.4\,\overline{\max(0,\lvert v_i-1\rvert-0.06)}+0.2\,\overline{(v_i-1)}$ | Voltage: smooth term + hard $\pm 0.06$ p.u. band exceedance + directional term |
| $g_T$ | $\overline{\max(0,\rho_k-1)}$ | Thermal overload of lines ($\rho$ = loading ratio) |
| $g_G$ | $0.5\,g^{P}_G+0.3\,g^{V}_G+0.2\,g^{C}_G$ | Mean relative amount the safety layer had to clip P, V and C commands |
| $g_C$ | $0.5\,g_{eff}+0.3\,g_{phys}+0.2\,g_{fair}$ | Curtailment composite (below) |

The curtailment composite has three parts:

- $g_{eff}=\max\!\big(0,\;\text{curtailed ratio}-\text{allowed ratio}\big)$, where the allowed ratio rises with congestion: $\text{clip}\big((\rho_{max}-0.9)/0.1,\,0,\,1\big)$. Curtail only when the grid actually needs it.
- $g_{phys}=\dfrac{\sum\max(0,\ \text{setpoint}-\text{actual output})}{\sum P^{avail}_{renewable}}$, the mismatch between the commanded and the realized renewable output.
- $g_{fair}=1-\text{Jain}(\text{curtailment ratios})$, so curtailment is spread fairly across plants.

### 4.5 Physics-informed graph: $Y_{bus}$ as edge features

At every step the environment estimates branch admittances from the Grid2Op observation, $y_\ell = I_\ell / (V_{or}-V_{ex})$ with $I_\ell = \overline{S_{or}/V_{or}}$, and assembles the bus admittance matrix

$$
Y_{bus}=G+jB
$$

- $G$ (conductance) captures active-power coupling and losses; $B$ (susceptance) captures reactive coupling and electrical distance.
- Both are clipped to $[-200,200]$, converted to magnitudes, standardized (z-score) and clipped to $\pm 3$, giving the 2-channel edge tensor $(\tilde G,\tilde B)$ of shape $[2,118,118]$.
- Entries of tripped lines are set to zero, so the graph is **topology-aware**: the attention layers see line outages immediately.

### 4.6 GAT actor-critic architecture

**Edge-conditioned attention (one layer, $H=4$ heads).** For head $h$:

$$
e^{h}_{ij}=\frac{\text{LeakyReLU}\!\Big(a_{src}^{h\top}Wh_i+a_{dst}^{h\top}Wh_j+\phi^{h}\big(\tilde G_{ij},\tilde B_{ij}\big)\Big)}{\sqrt{d_k}},\qquad
\alpha^{h}_{ij}=\text{softmax}_j\big(e^{h}_{ij}\big)
$$

$$
h_i' = \big\Vert_{h=1}^{H}\ \sum_{j}\alpha^{h}_{ij}\,W^{h}h_j
$$

where $\phi^{h}$ is the $h$-th output of a small edge MLP (Linear 2→128, ReLU, Linear 128→4) that turns the $(\tilde G,\tilde B)$ pair into one attention bias per head. Edges whose two channels are both exactly zero (tripped lines) are masked out before the softmax. Attention dropout is 0.1.

**Network layout (`GAT_ActorCritic`).**

```
Bus features [B,118,6] ─ per-graph standardize ─ Linear(6→32) + ReLU ─ + sinusoidal positional encoding
Generator features [B,248] ─ Linear(248→128) ─ ReLU ─ Linear(128→64) ─ broadcast to all 118 buses
          concat → node features [B,118,96]
          │
          ├─ residual projection Linear(96→128)
          ├─ GAT layer 1 (96→128, 4 heads) → LayerNorm → ReLU(+ residual)      = h1
          ├─ GAT layer 2 (128→128, 4 heads) → LayerNorm → ReLU(+ h1)           = h2
          │
          ├─ pooling: mean and max over buses of concat[h1,h2] → 512-dim
          ├─ fusion: Linear(512→256) + ReLU  → graph embedding h
          │
          ├─ gate:    g = sigmoid(Linear(256→1)(stop_grad(h)))
          ├─ Actor P: mu_P = sigmoid(Linear(256→62)(h) + cluster bias)
          ├─ Actor V: mu_V = g·sigmoid(Linear(318→62)([h, stop_grad(mu_P)]) + bias) + (1−g)·0.5
          ├─ Actor C: mu_C = g·sigmoid(Linear(318→62)([h, stop_grad(mu_P)]) + bias) + (1−g)·0.5
          └─ Critic:  V(s) = Linear(256→1)(h)
```

- **Policy distribution:** independent Gaussians per action dimension, $a\sim\mathcal{N}(\mu,\sigma^2)$, with learned state-independent $\sigma$ (initialized at $e^{-1.0}, e^{-1.5}, e^{-1.0}$ for P, V, C) and clamped to $[0.01, 0.35]$. Sampled actions are clipped to $[0,1]$ and the log-probability is computed on the clipped action.
- **Gate:** when the grid looks stable ($g\to0$) the voltage and curtailment heads fall back to the neutral value 0.5; under stress ($g\to1$) the learned emergency actions apply in full.
- **Generator broadcast:** every bus node sees global generation capacity, renewable availability and headroom, which is what makes coordinated dispatch possible with a message-passing model.
- **Size:** 271,270 trainable parameters.

### 4.7 Spatial clustering and soft attention masking

- **Offline zoning:** K-Means ($k=8$, `random_state=42`) on the rows of $\lvert Y\rvert=\sqrt{G^2+B^2}$ (diagonal zeroed) partitions the 118 buses into electrically coherent zones. Each generator inherits the cluster of its bus. This is computed once from the nominal topology.
- **Online mask:** buses with $\lvert V-1\rvert > 0.05$ p.u. flag their clusters. Generators get a mask value $m_k = 0.8$ if their cluster is flagged and $0.2$ otherwise. If no bus is flagged, $m_k=1$ for all generators.
- **Soft bias:** the mask enters the actor logits before the sigmoid as $b_k = 2\,(m_k-0.5)$, i.e. $+0.6$ (affected zone), $-0.6$ (unaffected zone), or a uniform $+1.0$ when nothing is violated.

$$
\mu_k=\text{sigmoid}\big(\text{Linear}(h)_k+b_k\big)
$$

The bias is soft (gradients still flow everywhere), so it steers exploration toward localized dispatch near the problem area without hard-restricting the action space.

### 4.8 Reward and PPO objective

Reward per step (no squashing, so failure gradients are preserved):

$$
r_t=W_{surv}+0.2\,W_{eq}\,J_t\,\sigma_t-\hat c_t-\sum_i\lambda_i g_i-\sum_i\frac{\rho_i}{2}g_i^2-W_{loss}\,\ell_t-W_{smooth}\,\Delta a_t-W_{redist}\,\overline{\Big(\tfrac{P^{tgt}-P^{act}}{P^{max}}\Big)^{2}}
$$

| Term | Meaning | Value |
|------|---------|-------|
| $W_{surv}$ | Bonus for every step survived | 2.0 |
| $J_t$ | Jain's fairness index of per-substation load satisfaction $q_k\in[0,1]$: $J=\frac{(\sum q_k)^2}{n\sum q_k^2}$ | weight $0.2\cdot W_{eq}$, $W_{eq}=2.0$ |
| $\sigma_t$ | Equity gate $=\text{clip}\big((\bar q-0.8)/0.2,0,1\big)$: fairness only pays when average satisfaction exceeds 80% | |
| $\hat c_t$ | Normalized generation cost $\sum_k(c_2P_k^2+c_1P_k)/\text{scale}$ | $c_2=0.01$, $c_1=0.05$ |
| $\ell_t$ | Loss ratio $(\sum P_{gen}-\sum P_{load})/\sum P_{load}$ | $W_{loss}=0.05$ |
| $\Delta a_t$ | Action smoothness: $1.0\,\text{MSE}_P+0.3\,\text{MSE}_V+0.1\,\text{MSE}_C$ vs. previous action | $W_{smooth}=0.05$ |
| Redispatch term | Penalizes commanded-vs-realized dispatch mismatch | $W_{redist}=0.3$ |
| Terminal | Early blackout: $-2.0\times$(fraction of episode remaining), applied after the first 200 training steps. Full episode completed: $+3.0$. | |

**PPO with GAE.**

$$
L^{CLIP}(\theta)=\mathbb{E}_t\Big[\min\big(r_t(\theta)A_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\big)\Big],\qquad r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad A_t=\sum_{l\ge0}(\gamma\lambda_{GAE})^{l}\,\delta_{t+l}
$$

$$
L_{total}=L_{actor}+\tfrac12\text{MSE}\big(V_\theta(s_t),\hat R_t\big)-\beta_{ent}\,H(\pi_\theta)
$$

| Hyperparameter | Value |
|----------------|-------|
| $\epsilon$ (clip) / $\gamma$ / $\lambda_{GAE}$ | 0.2 / 0.99 / 0.95 |
| Rollout length before each PPO update | 512 environment steps (not tied to episode boundaries) |
| Epochs / minibatch size | 5 / 64 |
| Optimizer / base LR / grad-clip | Adam / $10^{-4}$ / 0.5 |
| KL early stop | leave the minibatch loop when mean $(\log\pi_{old}-\log\pi_{new})>0.05$ |
| Entropy coefficient | $\beta_{ent}=\max(0.001,\ 0.01\,(1-k/1500))$, $k$ = PPO update index |
| LR schedule | `MultiStepLR`, milestones at PPO updates 100, 160, 220, factor 0.5 (LR: $10^{-4}\to5\!\times\!10^{-5}\to2.5\!\times\!10^{-5}\to1.25\!\times\!10^{-5}$) |

Returns are standardized with a running mean/variance (Welford's online algorithm) and clipped to $\pm3$, so the critic sees a consistent target scale even though raw returns differ by orders of magnitude between calm and failing episodes. Advantages are normalized per batch.

### 4.9 Dual (Lagrange multiplier) update

After each rollout, before the PPO step, the batch-mean violations $\bar g_i$ drive a projected dual ascent with decay:

$$
\lambda_i\leftarrow\min\!\Big(\lambda_{max},\ \max\big(0,\ (\lambda_i+\alpha\,\bar g_i)\cdot\gamma_{decay}\big)\Big),\qquad \alpha=10^{-3},\ \lambda_{max}=50,\ \gamma_{decay}=0.995
$$

The multipliers live in the trainer and are copied into the environment at the start of each episode, where they scale the penalty in the reward. The decay prevents windup: if violations disappear, the penalties relax instead of punishing the agent for past mistakes.

---

## 5) Curriculum training: the episode waves

Training runs **1,500 episodes** in five waves of increasing Suez gas-supply disruption. At the start of each episode the shock $s$ is drawn uniformly from the wave's range and applied to the gas/oil fleet as $a_P\leftarrow a_P\,(1-s)$. The wave changes automatically once the cumulative episode count of the wave is reached.

| Wave | Episodes (index range) | Shock range | Purpose |
|------|------------------------|-------------|---------|
| **I: Baseline** | 400 (0 to 399) | 0% | Learn nominal dispatch, warm up the multipliers |
| **II: Adaptation** | 300 (400 to 699) | 10% to 30% | Learn constraint-aware dispatch under moderate loss of fossil supply |
| **IIb: Stability Edge** | 300 (700 to 999) | 40% to 50% | Handle severe disruptions close to the stability limit |
| **III: Resiliency** | 400 (1000 to 1399) | 30% to 60% | Survive large capacity deficits while respecting constraints |
| **IV: Generalization** | 100 (1400 to 1499) | 0% to 70% (random) | Mixed regimes to consolidate robust behavior |

**How one training iteration works**

```
for each episode:
    sample shock from the current wave → set env.suez_shock_pct
    copy λ_V, λ_T, λ_G, λ_C from trainer to env
    roll out the stochastic policy until the episode ends (blackout or end of scenario)
        └─ every 512 collected steps (across episode boundaries):
              1. batch-mean g_V, g_T, g_G, g_C  →  dual ascent on λ
              2. GAE advantages + normalized returns
              3. PPO update (5 epochs, minibatch 64) + LR scheduler step
    log reward, equity, violations, g_i, λ_i, steps survived, latency
    save a checkpoint every 250 episodes
```

**Model selection (early stopping).** From episode 100 onward and only once training reaches Wave III (episode 1000), each episode is scored with a combined metric. A new best score (improvement > 0.01) saves `./checkpoints/best_gat_ppo_egypt118.pth`; training stops if there is no improvement for 200 episodes.

Periodic checkpoints are written to `./checkpoints/ckpt_step_{N:06d}.pth`. Training curves (reward, equity, $g_i$, $\lambda_i$, learning rate, losses) are stored in `output/output_csv/training_history.csv`.

---

## 6) Inference and evaluation

### 6.1 What happens at every control step

1. **Observe:** Grid2Op returns the state. The wrapper builds `bus_feat [118×6]`, `gen_feat [248]` and the $Y_{bus}$ edge tensor `[1,2,118,118]` for the current topology.
2. **(Optional) Spatial mask:** compute the generator cluster mask from buses with $\lvert V-1\rvert>0.05$ p.u. (Section 4.7).
3. **Forward pass:** the GAT actor-critic returns the P distribution, the V/C distribution and the value estimate.
4. **Choose the action:**
   - *deterministic* (recommended for deployment): use the distribution means;
   - *stochastic*: sample from the distributions (this is what the notebook's `evaluate_agent` helper does, and what training uses).
   Clip to $[0,1]$.
5. **Act:** the safety layer clips commands to generator limits, then Grid2Op applies redispatch, voltage setpoints and curtailment, and LightSim2Grid solves the AC power flow.
6. **Repeat** until the scenario ends or the grid blacks out.

The trained model is small (271K parameters) and runs on CPU; the per-step latency (policy forward pass plus simulation step) is recorded in the evaluation table as `latency_ms_avg`.

### 6.2 Minimal inference example (notebook classes)

```python
import numpy as np
import torch

# 1) Environment (the Grid2Op env must be created and seeded before the clusterer)
env = RobustEnv(noise_std=0.02, failure_prob=0.0, suez_shock_pct=0.30, is_test_env=True)
env.g2op_env.seed(42)

# 2) Optional: spatial clusterer (same settings as training)
clusterer = BusClusterer(env, n_clusters=8)

# 3) Model + weights
model = GAT_ActorCritic(env.n_bus, env.gen_count)
model.load_state_dict(torch.load("./checkpoints/best_gat_ppo_egypt118.pth",
                                 map_location="cpu", weights_only=True))
model.eval()

# 4) Control loop (deterministic)
(bus, gen), _ = env.reset()
done, total_reward = False, 0.0
with torch.no_grad():
    while not done:
        st_bus = torch.FloatTensor(bus).unsqueeze(0)
        st_gen = torch.FloatTensor(gen).unsqueeze(0)

        v_dev = np.abs(np.clip(env._get_bus_voltages(env._last_obs), 0.8, 1.2) - 1.0)
        cmask = torch.FloatTensor(
            clusterer.get_gen_mask_for_violations(v_dev, threshold=0.05)).unsqueeze(0)

        dist_P, dist_emg, value = model(st_bus, adj=env._y_bus_adj,
                                        gen_feat=st_gen, cluster_mask=cmask)
        action = torch.cat([dist_P.mean, dist_emg.mean], dim=-1).clamp(0, 1).numpy().flatten()

        (bus, gen), reward, done, _, info = env.step(action)
        total_reward += reward

print("Steps survived:", env.n_steps, "| reward:", round(total_reward, 1))
```

**Things to keep in mind**

- The K-Means zones are **not stored in the checkpoint**. Re-create the clusterer the same way as in training (same seed, same environment) or save `clusterer.bus_cluster` alongside your weights.
- The notebook's `evaluate_agent` calls the model without `cluster_mask` (the bias is off). If you want inference to match training exactly, pass the mask as above.
- Weights are loaded with `weights_only=True`; only load checkpoints you trust.
- The library file `smart_grid_gatpo_lib.py` packages the same stages (admittance builder, bus clustering, Grid2Op/pandapower adapters, and the `GATPO` controller); see its docstrings for exact signatures.

### 6.3 Evaluation protocol in the notebook

| Test | Setup |
|------|-------|
| **In-distribution test** | 500 episodes, shock drawn from 15% to 65%, observation noise $\sigma=0.02$, random-trip probability 0.05 |
| **Out-of-distribution (OOD)** | 300 episodes with harsher conditions: observation noise $\sigma=0.05$, random-trip probability 0.25, and a hook for shocking a held-out fossil generator subset (indices 20 to 29; `shock_percentage` is `0.0` by default in the notebook) |
| **Rule-based baselines** | Pro-Rata (scale by headroom), Merit-Order (renewables, then gas, then oil), Voltage-Sensitivity (favor generators near violated buses); 100 episodes each, same physics, reward and metrics |
| **Ablation** | Same GAT with the $Y_{bus}$ adjacency disabled (`GAT_ActorCritic_NoAdj`), trained and evaluated the same way |

Recorded metrics per episode: total reward, Jain equity (%), steps survived, voltage violations, max voltage deviation, max line loading, power loss, $g_V,g_T,g_G,g_C$, and latency. Plots (training dashboard, voltage heatmap before blackout, critic vs. actual return, attention heatmap, radar chart, sensitivity analysis, topology view) are saved to `output/` as HTML (Plotly) and PNG.

---

## 7) Repository structure

- `smart_grid_gatpo_lib.py` — reusable library implementation.
- `Smart_Grid_Agent.ipynb` — full experiment notebook (method + training + evaluation).
- `environment.yml` — (optional, see Section 1, Option B2) conda environment definition.
- `env_needed_csv/egypt_power_plants_processed.csv` — plant capacity/source data used by the environment logic.
- `checkpoints/` — model checkpoints.
- `output/` — generated plots and result files (`output/output_csv/` holds the CSV logs).

---

## 8) Notes

- The notebook is research-oriented and includes extensive diagnostics/plots.
- The library file is better for integration into production-like Python workflows.
- The agent is trained on the `l2rpn_wcci_2022` grid only; do not assume the results transfer to other grids or topologies without retraining.
