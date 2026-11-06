"""
proactive_hazard_routing/app.py
================================
Interactive Dashboard for Proactive Hazard-Adaptive Indoor Routing.
Compares the Baseline Greedy Engine (Paper 2) against the Proactive Dynamic Replanning Engine (Track A).
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2

# Path management
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'paper'))

import simulation as sim
import reviewer_experiments as rx
from src.baseline_evaluator import load_ground_floor
from src.proactive_engine import ProactiveCrowdSimulationEngine, ProactiveEvacuationAgent
from src.config import SimulationConfig, ProactiveRoutingConfig, BenchmarkLocations

st.set_page_config(page_title="Proactive Hazard Routing - Benchmark & Sim", layout="wide")

st.title("🛡️ Proactive Hazard Routing: Dynamic Indoor Evacuation")
st.markdown("""
**Mitigating Chokepoint Entrapment in Corridor Fires:**
The baseline crowd simulation engine relies on greedy local 8-neighborhood avoidance, which leads to ~45% occupant entrapment 
during critical corridor fires due to descent along stale static distance potentials ($\Phi^*(x)$).
This dashboard demonstrates **Proactive Dynamic Planning (Track A)**, verifying downstream trajectories ($H$),
repelling from flame buffers, and incrementally repairing paths to open alternate corridors.
""")

# Load floor data
@st.cache_data
def get_floor_data():
    img, free_space, graph, exits, annotations = load_ground_floor()
    return img, free_space, graph, exits, annotations

try:
    img, free_space, graph, exits, annotations = get_floor_data()
except Exception as e:
    st.error(f"Failed to load floor assets: {e}")
    st.stop()

locs = BenchmarkLocations()

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    num_agents = st.slider("Number of Occupants", min_value=20, max_value=250, value=100, step=10)
    
    st.subheader("Demographic Composition")
    c1, c2, c3 = st.columns(3)
    with c1: a_wt = st.number_input("Adult", 1, 10, 6)
    with c2: y_wt = st.number_input("Young", 0, 10, 3)
    with c3: e_wt = st.number_input("Elderly", 0, 10, 1)
    age_ratio = (a_wt, y_wt, e_wt)

    st.subheader("🔥 Fire Configuration")
    fire_preset = st.selectbox(
        "Chokepoint Preset",
        ["Origin A (Corridor Chokepoint: 250, 580)", "Origin B (West Wing Corridor: 175, 175)", "Custom"]
    )
    if "Origin A" in fire_preset:
        fx, fy = locs.origin_a_chokepoint
    elif "Origin B" in fire_preset:
        fx, fy = locs.origin_b_west
    else:
        fx = st.number_input("Fire X", 0, free_space.shape[1], 250)
        fy = st.number_input("Fire Y", 0, free_space.shape[0], 580)

    st.subheader("🧭 Proactive Search Parameters")
    lookahead = st.slider("Lookahead Horizon (H waypoints)", min_value=5, max_value=60, value=25, step=5)
    buffer_margin = st.slider("Hazard Buffer Margin (pixels)", min_value=2.0, max_value=30.0, value=15.0, step=1.0)
    sim_seed = st.number_input("Random Seed", value=0, step=1)

tabs = st.tabs(["📊 Comparative Benchmark", "🎬 Visual Simulation Walkthrough", "🧠 Algorithm Architecture"])

# TAB 1: Comparative Benchmark
with tabs[0]:
    st.subheader("Baseline Greedy vs. Proactive Routing Benchmark")
    st.markdown("Executes both engines under identical occupant spawn points and stochastic fire conditions.")

    if st.button("🚀 Run Comparative Benchmark", use_container_width=True):
        with st.spinner("Simulating Baseline (Greedy Local Avoidance)..."):
            rx._seed(sim_seed)
            t0 = time.time()
            h_base, ticks_base, avoid_base = sim.run_headless_experiment_v2(
                graph, free_space, exits, num_agents, age_ratio,
                fire_mode='dynamic', fire_start=(fx, fy),
                fire_spread_chance=0.08, fire_interval=5
            )
            t_base_elapsed = time.time() - t0
            evac_base = h_base[-1] if h_base else 0
            trapped_base = num_agents - evac_base

        with st.spinner("Simulating Proactive Hazard-Adaptive Engine..."):
            rx._seed(sim_seed)
            t0 = time.time()
            sim_cfg = SimulationConfig(num_agents=num_agents, age_ratio=age_ratio)
            rout_cfg = ProactiveRoutingConfig(lookahead_waypoints=lookahead, hazard_buffer_margin=buffer_margin)

            engine_pro = ProactiveCrowdSimulationEngine(
                graph, free_space, exits, sim_config=sim_cfg, routing_config=rout_cfg,
                fire_mode='dynamic', fire_start=(fx, fy)
            )
            # Spawn agents identical to baseline
            active_paths = engine_pro.paths
            valid_spawn = [node for node in active_paths.keys() if graph.nodes[node].get('type') == 'free']
            spawn_points = random.sample(valid_spawn, min(num_agents, len(valid_spawn)))

            a_pct, y_pct, e_pct = age_ratio[0] / sum(age_ratio), age_ratio[1] / sum(age_ratio), age_ratio[2] / sum(age_ratio)
            types = ['adult'] * int(num_agents * a_pct) + ['young'] * int(num_agents * y_pct) + ['elderly'] * int(num_agents * e_pct)
            while len(types) < len(spawn_points):
                types.append('adult')
            random.shuffle(types)

            for i, s_node in enumerate(spawn_points):
                path_nodes = active_paths[s_node][::-1]
                coords = [graph.nodes[n]['pos'] for n in path_nodes]
                agent = ProactiveEvacuationAgent(
                    agent_id=i, start_pos=coords[0], path=coords, agent_type=types[i],
                    lookahead=lookahead, sensing_radius=40.0
                )
                engine_pro.add_agent(agent)

            h_pro = []
            while not engine_pro.is_complete():
                engine_pro.step()
                h_pro.append(engine_pro.evacuated_count)

            t_pro_elapsed = time.time() - t0
            evac_pro = engine_pro.evacuated_count
            trapped_pro = num_agents - evac_pro

        # Metrics Summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Evacuated", f"{evac_base}/{num_agents}", f"{(evac_base/num_agents)*100:.1f}%")
        m2.metric("Proactive Evacuated", f"{evac_pro}/{num_agents}", f"{(evac_pro/num_agents)*100:.1f}%", delta=f"+{((evac_pro-evac_base)/num_agents)*100:.1f}%")
        m3.metric("Baseline Trapped", f"{trapped_base}", delta=f"{trapped_base} trapped", delta_color="inverse")
        m4.metric("Proactive Trapped", f"{trapped_pro}", delta=f"-{trapped_base - trapped_pro}", delta_color="normal")

        # Evacuation Curves Plot
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.plot(range(len(h_base)), h_base, label="Baseline (Greedy Avoidance)", color="#d9534f", linewidth=2.5)
        ax.plot(range(len(h_pro)), h_pro, label="Proactive Hazard Routing (Lookahead H)", color="#28a745", linewidth=2.5)
        ax.axhline(num_agents, color="gray", linestyle="--", alpha=0.6, label="Total Occupants")
        ax.set_xlabel("Simulation Ticks", fontsize=11)
        ax.set_ylabel("Cumulative Evacuated Occupants", fontsize=11)
        ax.set_title(f"Evacuation Egress Dynamics at Chokepoint ({fx}, {fy})", fontsize=13, fontweight='bold')
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True)
        st.pyplot(fig)

        # Comparative Table
        df_comp = pd.DataFrame([
            {
                "Engine": "Baseline Greedy",
                "Evacuated (%)": f"{(evac_base/num_agents)*100:.1f}%",
                "Occupants Trapped": trapped_base,
                "Completion Ticks": ticks_base,
                "Runtime (s)": f"{t_base_elapsed:.2f}s",
                "Avoidance Rate": f"{avoid_base*100:.1f}%",
            },
            {
                "Engine": "Proactive Routing (Track A)",
                "Evacuated (%)": f"{(evac_pro/num_agents)*100:.1f}%",
                "Occupants Trapped": trapped_pro,
                "Completion Ticks": engine_pro.current_tick,
                "Runtime (s)": f"{t_pro_elapsed:.2f}s",
                "Avoidance Rate": f"{engine_pro.avoidance_fraction()*100:.1f}%",
            }
        ])
        st.dataframe(df_comp, use_container_width=True)

# TAB 2: Visual Simulation Walkthrough
with tabs[1]:
    st.subheader("Floor Plan Inspection & Fire Front Geometry")
    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("**Navigable Space with Chokepoint Location**")
        disp_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        cv2.circle(disp_img, (int(fx), int(fy)), 12, (0, 0, 255), -1)
        cv2.putText(disp_img, f"Fire Chokepoint ({fx},{fy})", (int(fx)+15, int(fy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
        
        for e in exits:
            epos = graph.nodes[e]['pos']
            cv2.rectangle(disp_img, (int(epos[0])-8, int(epos[1])-8), (int(epos[0])+8, int(epos[1])+8), (0, 200, 0), -1)
            cv2.putText(disp_img, "Exit", (int(epos[0])+12, int(epos[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

        st.image(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB), caption="Ground Floor with Egress Chokepoint and Exit", use_container_width=True)

    with colB:
        st.markdown("**Key Algorithmic Mechanism**")
        st.info("""
        **Why Baseline Fails:**
        - Distance map $\Phi^*(x)$ is calculated only once at $t=0$.
        - When fire consumes the corridor at `(250, 580)`, moving backward increases $\Phi^*(x)$.
        - Because greedy avoidance only searches the immediate 8-neighborhood $[-2, 0, 2]$, agents refuse to step uphill ($\Delta \Phi^* > 0$) to take the parallel corridor, trapping 45% of occupants.

        **How Proactive Engine Solves It:**
        - Lookahead horizon ($H=25$) projects future waypoints along trajectory.
        - Hazard buffer via Euclidean Distance Transform repels agents before contact.
        - Event-driven path repair dynamically invalidates compromised edges and steers agents into clear alternate corridors.
        """)

# TAB 3: Algorithm Architecture
with tabs[2]:
    st.subheader("System Architecture & Roadmap")
    st.markdown("""
    ### Dual-Track Architecture for Autonomous Hazard Avoidance
    
    1. **Track A: Dynamic Incremental Search (Operational)**
       - Trajectory projection across horizon $H$.
       - Euclidean Distance Transform repulsion buffer around expanding flame front.
       - Event-driven graph repair bypassing burned corridors.
       
    2. **Track B: Multi-Agent Reinforcement Learning (MARL Environment)**
       - Gymnasium multi-agent environment wrapping architectural raster.
       - Multi-channel observation tensor $(21 \\times 21 \\times 6)$: Free space, Flame, Smoke Buffer, Crowd Density, Exit Gradient, Demographic Priority.
       - Parameter-shared PPO / MAPPO with centralized critic.
    """)
