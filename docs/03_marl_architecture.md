# 🤖 Track B: Multi-Agent Reinforcement Learning (MARL) Architecture

## 1. Formal Formulation: Dec-POMDP

The crowd evacuation problem with dynamic hazards is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) defined by the tuple:

$$\mathcal{M}_{\text{Dec-POMDP}} = \langle \mathcal{N}, \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

* $\mathcal{N} = \{1, 2, \dots, N\}$: Set of heterogeneous crowd occupants.
* $\mathcal{S}$: Global state space containing all agent positions $\mathbf{p}_{1:N}$, demographic types $c_{1:N}$, occupancy grid $C$, and fire state $\mathcal{F}$.
* $\mathcal{A}$: Discrete action space available to each agent.
* $\mathcal{P}(s' \mid s, \mathbf{a})$: Environmental transition dynamics combining agent movement, stochastic fire propagation, and demographic tick scheduling.
* $\mathcal{R} = \{R_1, \dots, R_N\}$: Individual agent reward functions.
* $\Omega$: Observation space for decentralized execution.
* $\mathcal{O}(s, i) \to o_i$: Observation emission function providing a local view to agent $i$.
* $\gamma \in [0.95, 0.99]$: Temporal discount factor.

---

## 2. Observation Space Design ($o_i$)

Each agent receives a localized egocentric tensor of size $K \times K \times C$ (default $21 \times 21 \times 6$ pixels, centered on the agent) along with a demographic vector:

### Multi-Channel Spatial Feature Map ($21 \times 21 \times 6$)
1. **Channel 0 (Traversability):** Binary mask of navigable floor space ($\mathcal{M}=1$) and walls ($\mathcal{M}=0$).
2. **Channel 1 (Crowd Density):** Binary mask of cells occupied by peer agents ($C(t)=1$).
3. **Channel 2 (Peer Demographics):** Continuous channel encoding peer priorities ($\pi_{\text{young}}=0.33, \pi_{\text{elderly}}=0.66, \pi_{\text{adult}}=1.0$).
4. **Channel 3 (Active Fire Front):** Binary mask of burning cells ($\mathcal{F}(t)=1$).
5. **Channel 4 (Hazard Repulsion Gradient):** Continuous Euclidean distance transform from the nearest flame.
6. **Channel 5 (Global Exit Potential):** Normalized static distance field $\Phi^*(\mathbf{x}) / \max \Phi^*$ indicating the nominal downhill escape gradient.

### Proprioceptive Vector ($\mathbb{R}^7$)
* One-hot demographic type: $[c_{\text{adult}}, c_{\text{young}}, c_{\text{elderly}}]$
* Normalized velocity / step counter: $[\text{ticks\_remaining} / \tau_i]$
* Distance to nearest exit: $[d_{\text{exit}} / D_{\text{max}}]$
* Distance to nearest fire: $[d_{\text{fire}} / D_{\text{max}}]$
* Unit vector pointing toward primary exit: $[\hat{u}_x, \hat{u}_y]$

---

## 3. Action Space ($\mathcal{A}$)

### Mode 1: Discrete Spatial Navigation (9-DoF)
$$\mathcal{A}_{\text{discrete}} = \{\text{Stay}, \text{North}, \text{North-East}, \text{East}, \text{South-East}, \text{South}, \text{South-West}, \text{West}, \text{North-West}\}$$
Each step attempts to translate the agent by step size (2 pixels) in the chosen direction. Invalid movements (into walls or occupied cells) are masked or rejected.

### Mode 2: Macro-Action / Goal-Conditioned Steering (Hierarchical MARL)
$$\mathcal{A}_{\text{macro}} = \{\text{Follow Default Path}, \text{Detour Left}, \text{Detour Right}, \text{Retreat to Room}, \text{Switch to Secondary Exit}, \text{Yield Right-of-Way}\}$$
High-level macro actions invoke sub-graph local planners, drastically shrinking the exploration horizon.

---

## 4. Multi-Objective Reward Shaping

$$\mathcal{R}_i(t) = R_{\text{evac}} + R_{\text{progress}} + R_{\text{hazard}} + R_{\text{collision}} + R_{\text{social}} + R_{\text{time}}$$

| Term | Mathematical Formulation | Weight / Value | Purpose |
| :--- | :--- | :--- | :--- |
| **$R_{\text{evac}}$** | $+R_{\text{goal}} \cdot \mathbb{I}(\text{reached exit})$ | $+100.0$ | Primary task incentive |
| **$R_{\text{progress}}$** | $\beta \cdot \max(0, \Phi^*(\mathbf{p}_i(t-1)) - \Phi^*(\mathbf{p}_i(t)))$ | $+2.0$ | Encourages forward egress flow |
| **$R_{\text{hazard}}$** | $-R_{\text{fatal}} \cdot \mathbb{I}(\mathbf{p}_i \in \mathcal{F}) - \kappa \cdot \exp(-d_i / \lambda)$ | $-200.0$ (burn), $\kappa=10.0$ | Strict proactive fire avoidance |
| **$R_{\text{collision}}$** | $-c_{\text{bump}} \cdot \mathbb{I}(\text{attempted collision})$ | $-1.5$ | Prevents crowding jams & cell contention |
| **$R_{\text{social}}$** | $+r_{\text{yield}} \cdot \mathbb{I}(\text{adult yielded to elderly})$ | $+3.0$ | Enforces demographic right-of-way |
| **$R_{\text{time}}$** | $-c_{\text{tick}}$ per simulation step | $-0.05$ | Minimizes total evacuation time |

---

## 5. Training Paradigm & Network Architecture

```
                                [ Centralized Critic V(s) ]
                                (Full Map State, Global Occupancy)
                                              ▲
                                              │ Centralized Training
                       ┌──────────────────────┴──────────────────────┐
                       │                                             │
             [ Policy Network π_θ ]                        [ Policy Network π_θ ]
             (Agent 1 Local Window)                        (Agent 2 Local Window)
                       │                                             │
                       ▼                                             ▼
                 Action a_1                                    Action a_2
                    (Decentralized Execution at Test Time)
```

* **Algorithm:** Multi-Agent Proximal Policy Optimization (**MAPPO**) with parameter sharing across agents of the same demographic class.
* **Centralized Training, Decentralized Execution (CTDE):** The Critic $V_\psi(s)$ observes global floor state during simulation training, while the Actor $\pi_\theta(a_i \mid o_i)$ executes purely on local sensory windows.
* **Action Masking:** Actions attempting to step into impassable walls or occupied cells are masked out with logits set to $-\infty$, preventing trivial invalid action exploration.
