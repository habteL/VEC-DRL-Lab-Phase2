# VEC-DRL-Lab-Phase2

**Multi-Vehicle Vehicular Edge Computing Simulator 
with Parameter-Sharing MARL Scheduling**

Extends Phase 1 (single-vehicle DQN) to a two-vehicle 
Multi-Agent Reinforcement Learning system with contention-aware 
scheduling and fairness optimization.

Phase 1 repository: https://github.com/habteL/VEC-DRL-Lab

## Author
**Dr. Habte Lejebo**  
Research Areas: Vehicular Edge Computing · Multi-Agent RL · 
Fog Computing · Deep Reinforcement Learning

## Key Results

| Policy | Completion Rate | Jain Index |
|---|---|---|
| Always Offload | 26.4% | 0.978 |
| Always Local | 20.3% | 0.990 |
| **MARL DQN Agent** | **46.0%** | **0.992** |

MARL DQN outperforms always-offload by **74.2%** while achieving 
near-perfect fairness (Jain=0.992) across two heterogeneous vehicles.

## What's New In Phase 2

| Feature | Phase 1 | Phase 2 |
|---|---|---|
| Vehicles | 1 | 2 (heterogeneous) |
| Agent architecture | Single DQN | Parameter-sharing MARL |
| State vector | 6 elements | 7 elements (+ contention) |
| Fairness metric | None | Jain's Index |
| Resource contention | None | Modeled + penalized |
| Handover migration | Single vehicle | Per-vehicle independent |

## System Configuration