# 📐 Mathematical Problem Formulation & Entrapment Anatomy

## 1. Environment & Graph Representation

The indoor environment is modeled as a 2-tuple $(\mathcal{M}, G)$ where:
1. $\mathcal{M} \in \{0, 1\}^{H \times W}$ is a high-resolution discrete occupancy grid representing physical boundaries, walls ($\mathcal{M}_{y,x}=0$), and navigable free space ($\mathcal{M}_{y,x}=1$).
2. $G = (V, E, w, \phi)$ is the constraint-aware semantic building graph:
   * $V = V_{\text{free}} \cup V_{\text{door}} \cup V_{\text{exit}} \cup V_{\text{stairs}} \cup V_{\text{lift}}$
   * $E \subseteq V \times V$ represents spatial traversability between 8-connected grid pixels or semantic waypoints.
   * $w: E \to \mathbb{R}_{>0}$ is the positive edge weight incorporating Euclidean distance, doorway discounts ($\alpha = 0.6$), and priority/maintenance modifiers.
   * $\phi: V \to \text{Type}$ is the architectural semantic typing function.

---

## 2. Agent Dynamics & Crowd Physics

A crowd of $N$ heterogeneous agents $\mathcal{A} = \{a_1, a_2, \dots, a_N\}$ navigates the environment. Each agent $a_i$ is characterized by:
* $\mathbf{p}_i(t) \in \mathbb{R}^2$: Spatial position at discrete simulation tick $t$.
* $c_i \in \{\text{adult}, \text{young}, \text{elderly}\}$: Demographic classification.
* $\tau_i \in \mathbb{Z}^+$: Ticks required per spatial step (inversely proportional to velocity):
  $$\tau_{\text{adult}} \sim \mathcal{U}\{2, 3\}, \quad \tau_{\text{young}} \sim \mathcal{U}\{3, 4\}, \quad \tau_{\text{elderly}} \sim \mathcal{U}\{4, 6\}$$
* $\pi_i \in \{1, 2, 3\}$: Bottleneck priority level (lower values yielded right-of-way):
  $$\pi_{\text{young}} = 1, \quad \pi_{\text{elderly}} = 2, \quad \pi_{\text{adult}} = 3$$

### Collision & Exclusive Cell Occupancy
Space exclusive occupancy is strictly enforced via a boolean grid $C(t) \in \{0, 1\}^{H \times W}$:
$$C_{y, x}(t) = 1 \iff \exists a_i \text{ such that } \lfloor \mathbf{p}_i(t) \rceil = (x, y)$$
No two agents may occupy the same cell simultaneously:
$$\forall a_i \neq a_j, \quad \lfloor \mathbf{p}_i(t) \rceil \neq \lfloor \mathbf{p}_j(t) \rceil$$

---

## 3. Dynamic Hazard Model: Stochastic Spreading Fire

Let $\mathcal{F}(t) \subset \mathbb{Z}^2$ denote the set of engulfed grid coordinates at tick $t$.
1. **Ignition:** At $t = 0$, fire ignites at origin $\mathbf{x}_{\text{fire}} = (x_0, y_0) \in V_{\text{free}}$, setting $\mathcal{F}(0) = \{\mathbf{x}_{\text{fire}}\}$.
2. **Stochastic Propagation:** Every $k$ ticks (default $k=5$), the active fire perimeter $\partial \mathcal{F}(t)$ attempts to infect unburned adjacent navigable cells $\mathbf{u} \in \mathcal{N}_4(\mathbf{v})$ with independent probability $\sigma$ (default $\sigma = 0.08$):
   $$\mathbf{u} \in \mathcal{F}(t+1) \iff \mathbf{u} \in \mathcal{N}_4(\mathbf{v}), \, \mathbf{v} \in \partial \mathcal{F}(t), \, \mathcal{M}_{\mathbf{u}} = 1, \, U(0, 1) < \sigma$$

---

## 4. The Anatomy of Chokepoint Entrapment (Why Baseline Fails)

In Paper 2, Experiment 2b, fire at a central corridor chokepoint resulted in **45.3% entrapment**:

$$\text{Evacuated Fraction} = 0.547 \pm 0.066, \quad \text{Mean Trapped} = 67.9 / 150$$

```
                                  [ CORRIDOR CHOKEPOINT TRAP ]
                                  
                       FIRE FRONT SPREADING
               ┌─────────────────▼─────────────────┐
   ════════════╧═══════════════════════════════════╧═══════════ Wall
   Room A                                                      Corridor Exit
   [Agents] ──► ──► ──► [ Dense Crowd Deadlock ] ──X (Engulfed) ──► [EXIT GOAL]
   ════════════════════════════════════════════════════════════ Wall
                        ▲
                        │ Agents cannot move backward against 
                        │ the static distance gradient!
```

### The Root Cause: Static Gradient Inversion
1. **Static Distance Potential:** Each agent computes its nominal trajectory $\mathcal{P}_i = (v_0, v_1, \dots, v_m)$ by descending the global static distance potential field:
   $$\Phi^*(\mathbf{x}) = \min_{e \in V_{\text{exit}}} d_G(\mathbf{x}, e)$$
2. **Local Myopic Avoidance Rule:**
   When the planned next node $v_{t+1} \in \mathcal{F}(t)$ is engulfed, the agent executes local search in its 8-neighborhood $\mathcal{N}_8(\mathbf{p}_i(t))$:
   $$\mathbf{p}^* = \arg\min_{\mathbf{u} \in \mathcal{N}_8(\mathbf{p}_i(t)) \setminus \mathcal{F}(t)} \Phi^*(\mathbf{u})$$
3. **The Dead-End Dilemma:**
   * Inside a corridor, $\nabla \Phi^*(\mathbf{x})$ vectors point *down the corridor toward the exit*.
   * Even when the fire completely bisects the corridor downstream, $\Phi^*$ values immediately in front of the fire remain lower than $\Phi^*$ values behind the agent.
   * Consequently, the local greedy rule **pulls agents directly into the flame front**, pressing them against the fire boundary.
   * Incoming agents from Room A continue walking forward, physically boxing in the front agents via exclusive cell occupancy $C(t)$.
   * **Escaping requires increasing the static distance $\Phi^*(\mathbf{x})$** (i.e., moving backward toward Room A, taking an alternate corridor, or redirecting to stairs). The greedy baseline cannot do this because it lacks:
     * Lookahead horizon $> 1$ step.
     * Dynamic path invalidation when downstream segments are blocked.
     * Multi-destination re-evaluation.
