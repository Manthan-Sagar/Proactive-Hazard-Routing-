# 🧭 Track A: Dynamic Incremental Search & Proactive Planning Framework

## 1. Overview & Principles

Track A provides a mathematically sound, reproducible, and explainable algorithmic solution. Rather than waiting until an agent collides with flames, Track A implements **predictive sensing, dynamic path invalidation, and incremental replanning**.

```
  [ Spreading Fire Event ] ──► [ Edge Invalidation: w(u,v) = ∞ ] ──► [ Incremental D* Lite Repair ]
                                                                             │
  [ Agent Downstream Sense ] ──► [ Path Severed Ahead? ] ───────► [ Proactive Turnaround / Alternate Exit ]
```

---

## 2. Proactive Sensing Horizon ($R_{\text{sense}}$)

Each agent $a_i$ maintains a sensory horizon with radius $R_{\text{sense}}$ (typically 30 px ≈ 0.6 m to 60 px ≈ 1.2 m):

### A. Sensory Modalities
1. **Euclidean / Metric Sensor:** 
   $$\mathcal{S}_{\text{metric}}(a_i) = \{\mathbf{x} \in \mathcal{F}(t) \mid \|\mathbf{p}_i(t) - \mathbf{x}\|_2 \le R_{\text{sense}}\}$$

2. **Downstream Route Interception (Lookahead Horizon $H$):**
   Instead of sensing isotropic space, the agent verifies the next $H$ waypoints along its planned trajectory $\mathcal{P}_i$:
   $$\text{BlockedAhead}(a_i) = \text{True} \iff \exists j \in [k, k + H] \text{ s.t. } \mathcal{P}_i[j] \in \mathcal{F}(t) \cup \mathcal{B}_{\text{hazard}}$$
   where $\mathcal{B}_{\text{hazard}}$ is the Euclidean distance transform safety buffer surrounding the fire front.

3. **Line-of-Sight (Bresenham Raycast):**
   Checks whether smoke or flame is visually observable without intervening wall occlusion:
   $$\text{Visible}(\mathbf{p}_i, \mathbf{x}_{\text{fire}}) = \text{BresenhamClear}(\mathbf{p}_i, \mathbf{x}_{\text{fire}}, \mathcal{M})$$

---

## 3. Dynamic Cost Field & Hazard Repulsion Buffer

To prevent agents from skimming along the edge of active flames, the graph edge cost function $w(u, v)$ is dynamically modified by a hazard repulsion potential:

$$w_{\text{dynamic}}(u, v, t) = \begin{cases}
\infty & \text{if } u \in \mathcal{F}(t) \text{ or } v \in \mathcal{F}(t) \\
w_{\text{base}}(u, v) \cdot \left[1 + \gamma \exp\left(-\frac{d(v, \mathcal{F}(t))}{\lambda}\right)\right] & \text{otherwise}
\end{cases}$$

Where:
* $d(v, \mathcal{F}(t))$: Euclidean distance from node $v$ to the nearest burning cell.
* $\lambda$: Exponential decay length scale (e.g., 15 px).
* $\gamma$: Maximum penalty multiplier (e.g., 10.0).

This continuous potential deflects agents away from corridors with encroaching smoke before flames physically touch them.

---

## 4. Incremental Replanning Algorithms

### Approach 1: Event-Driven Dynamic Multi-Source Dijkstra
* **Trigger:** Whenever fire expands into a new corridor or an agent senses that its trajectory is severed ($\text{BlockedAhead} = \text{True}$).
* **Execution:**
  1. Remove or set weight $\infty$ for edges touching engulfed cells $\mathcal{F}(t)$.
  2. Compute reverse multi-source Dijkstra from all unblocked exits:
     $$\text{dist\_map}(v) = \min_{e \in V_{\text{exit}} \setminus \mathcal{F}(t)} d_{G_t}(v, e)$$
  3. Affected agents obtain updated global paths immediately.
* **Complexity:** Fast on grid graphs with cached components (< 25 ms on 50,000 nodes using binary heaps).

### Approach 2: D* Lite for High-Density Scaling
For scenarios with continuous, high-frequency fire spread:
* Each node $u$ maintains $g(u)$ (current cost estimate) and $rhs(u)$ (one-step lookahead cost):
  $$rhs(u) = \begin{cases} 0 & \text{if } u \in V_{\text{exit}} \\ \min_{v \in \mathcal{N}(u)} (g(v) + c(u, v)) & \text{otherwise} \end{cases}$$
* Nodes are inconsistent if $g(u) \neq rhs(u)$.
* Priority queue maintains inconsistent nodes with key:
  $$k(u) = \left[\min(g(u), rhs(u)) + h(s_{\text{start}}, u) + k_m, \, \min(g(u), rhs(u))\right]$$
* Only the sub-graph affected by expanding fire cells is updated, saving significant computation compared to full replanning.

---

## 5. Topological Exit & Corridor Switching Logic

When an agent in a corridor detects a fire front blocking the corridor exit:
1. **Immediate State Transition:** Switch agent state from `NORMAL_EGRESS` to `CORRIDOR_EVASION`.
2. **Reverse Target Waypoint:** Identify the nearest topological junction or room center behind the agent.
3. **Alternate Destination Routing:** Reroute toward the secondary exit or nearest fire stairs ($V_{\text{stairs}}$).
4. **Collision Cooperation:** High-priority agents signal backward evacuation, causing trailing occupants to halt or retreat cooperatively.
