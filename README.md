```markdown
# VEC-DRL-Lab-Phase2

**Multi-Vehicle Vehicular Edge Computing Simulator  
with Parameter-Sharing MARL Scheduling**

Extends [Phase 1](https://github.com/habteL/VEC-DRL-Lab) 
(single-vehicle DQN) to a two-vehicle Multi-Agent Reinforcement 
Learning system with contention-aware scheduling, fairness 
optimization, load regime analysis, and scalability evaluation.

---

## Author

**Dr. Habte Lejebo**  
Research Areas: Vehicular Edge Computing · Multi-Agent RL ·  
Fog Computing · Deep Reinforcement Learning

---

## Research Questions Answered

| Question | Finding |
|---|---|
| C1: Can MARL handle multi-vehicle contention? | ✅ Yes — two vehicles, Jain=0.992 |
| C2: Does parameter sharing improve fairness? | ✅ Structural fairness by design |
| C3a: Does it outperform simple strategies? | ✅ +74.2% over always-offload |
| C3b: Does it adapt to traffic intensity? | ✅ Graceful degradation across regimes |
| C3c: Does it learn fairness? | ✅ Fairness is architectural, not learned |
| C3d: Does it scale to more vehicles? | ✅ Partial — capacity-limited, not policy-limited |

---

## Key Results

### Baseline Comparison (C3a)

| Policy | Completion Rate | Jain Index |
|---|---|---|
| Always Offload | 26.4% | 0.978 |
| Always Local | 20.3% | 0.990 |
| **MARL DQN Agent** | **46.0%** | **0.992** |

MARL DQN outperforms always-offload by **74.2%** while achieving
near-perfect fairness (Jain=0.992) across two heterogeneous vehicles.

### Load Regime Analysis (C3b)

| Regime | Arrival Rate | Completion Rate |
|---|---|---|
| Low | randint(0,1) | 70.1% |
| Medium | randint(0,2) | 45.5% |
| High | randint(0,3) | 46.0% |

Performance degradation from medium to high is minimal — the agent
demonstrates graceful degradation under increasing congestion.

### Scalability Analysis (C3d)

| Configuration | Completion Rate | Jain Index |
|---|---|---|
| 2 vehicles (trained) | 46.0% | 0.992 |
| 3 vehicles (zero-shot) | 38.1% | 0.980 |
| 3 vehicles (retrained) | 37.5% | 0.981 |

Performance degradation with 3 vehicles is capacity-driven
(ρ increases from 3.6 to 5.4), not policy-driven. Fairness is
maintained across all configurations.

---

## What's New In Phase 2

| Feature | Phase 1 | Phase 2 |
|---|---|---|
| Vehicles | 1 | 2–3 (heterogeneous) |
| Agent architecture | Single DQN | Parameter-sharing MARL |
| State vector | 6 elements | 7 elements (+ contention) |
| Fairness metric | None | Jain's Index |
| Resource contention | None | Modeled + penalized |
| Load regime analysis | None | Low / Medium / High |
| Scalability study | None | Zero-shot + retrained |

---

## System Configuration

```
Vehicle 1: speed=2, start x=0   (baseline vehicle)
Vehicle 2: speed=1, start x=15  (full 35-step coverage)
Vehicle 3: speed=1.5, start x=5 (C3d only)

Server 1: x=10, capacity=300, radius=10
Server 2: x=30, capacity=300, radius=10
Server 3: x=50, capacity=300, radius=10

Contention window (2V): steps 10-20 on Server 2
Contention window (3V): steps 0-5  on Server 1
                        steps 10-15 on Server 2
```

---

## Key Research Findings

**Finding 1 — Resource contention cascade:**
Queue buildup during the contention window (steps 10-20) propagates
to Server 3 through backhaul migration, causing secondary congestion
even after vehicles separate.

**Finding 2 — Structural fairness via parameter sharing:**
Parameter-sharing MARL provides fairness guarantees by design.
Jain index stays near 1.0 from episode 1 — fairness is architectural,
not an emergent learned behavior.

**Finding 3 — Graceful degradation under load:**
MARL maintains stable completion rates across medium and high load
regimes. Performance bounds are capacity-limited (ρ >> 1), not
algorithm-limited.

**Finding 4 — Capacity-limited scalability:**
Zero-shot 3-vehicle deployment reduces completion by 17.2% due to
increased utilization (ρ: 3.6 → 5.4), not policy failure.
Retraining provides minimal additional improvement.

---

## Project Structure

```
VEC-DRL-Lab-Phase2/
├── src/vecsim/
│   ├── task.py
│   ├── vehicle.py           # + current_server_id, arrival_max param
│   ├── edge_server.py       # + completed_tasks buffer
│   ├── channel.py
│   └── agent.py             # state_size parameterized
├── experiments/
│   ├── simulation.py        # MARL training (--arrival-max, --regime)
│   ├── baseline_comparison_marl.py
│   ├── plot.py              # learning curve
│   ├── plot_regimes.py      # three load regimes
│   ├── plot_fairness.py     # per-vehicle fairness dynamics
│   └── scalability_c3d.py   # three-vehicle scalability
├── results/
│   ├── comparison_marl.png
│   ├── learning_curve_marl.png
│   ├── regime_comparison.png
│   ├── fairness_dynamics.png
│   ├── scalability_c3d.png
│   ├── trained_agent_marl_low.npy
│   ├── trained_agent_marl_medium.npy
│   ├── trained_agent_marl_high.npy
│   └── trained_agent_3vehicles.npy
└── docs/
    └── VEC_Lab_Manual.pdf
```

---

## Quick Start

```bash
git clone https://github.com/habteL/VEC-DRL-Lab-Phase2.git
cd VEC-DRL-Lab-Phase2
pip install -e .

# Train MARL agent (high load)
python experiments/simulation.py --arrival-max 3 --regime high

# Run baseline comparison
python experiments/baseline_comparison_marl.py

# Plot learning curves
python experiments/plot.py

# Load regime analysis
python experiments/simulation.py --arrival-max 1 --regime low
python experiments/simulation.py --arrival-max 2 --regime medium
python experiments/plot_regimes.py

# Fairness dynamics
python experiments/plot_fairness.py

# Three-vehicle scalability
python experiments/scalability_c3d.py
```

---

## Acknowledgements

Developed with AI-assisted mentoring (Claude, Anthropic).
All research contributions, design decisions, and implementation
by the author.

## Citation

```bibtex
@misc{lejebo2026vecmarl,
  author    = {Lejebo, Leka Habte},
  title     = {{VEC Research Laboratory Phase 2:
               Multi-Vehicle MARL Scheduling}},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/habteL/VEC-DRL-Lab-Phase2}
}
```

See [CITATION.md](CITATION.md) for related IEEE publications.

## License
- Source code: MIT License
- Documentation: CC BY 4.0
```