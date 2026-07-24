import csv
import random
import math
import numpy as np
from vecsim.vehicle     import Vehicle
from vecsim.edge_server import EdgeServer
from vecsim.channel     import WirelessChannel
from vecsim.agent       import DQNAgent

random.seed(42)
np.random.seed(42)

STEP_DURATION  = 0.1
NUM_STEPS      = 35
NUM_EPISODES   = 1000

# ── Two heterogeneous vehicles ────────────────────────────────
vehicles = [
    Vehicle(vehicle_id=1, x=0,  y=0, speed=2, direction=0),
    Vehicle(vehicle_id=2, x=15, y=0, speed=1, direction=0),
]

# ── Three servers ─────────────────────────────────────────────
servers = {
    1: EdgeServer(server_id=1, capacity=300, x=10, y=0,
                  coverage_radius=10),
    2: EdgeServer(server_id=2, capacity=300, x=30, y=0,
                  coverage_radius=10),
    3: EdgeServer(server_id=3, capacity=300, x=50, y=0,
                  coverage_radius=10),
}

channel = WirelessChannel(
    bandwidth=5_000_000,
    transmission_power=1000,
    noise_factor=1
)

# ── Shared DRL agent — state_size=7 ──────────────────────────
agent = DQNAgent(state_size=7, action_size=2)

episode_metrics     = []
pending_experiences = {}

print("=== Multi-Vehicle MARL Training ===")
print(f"Vehicles: 2  |  Servers: 3  |  Episodes: {NUM_EPISODES}")
print()

for episode in range(NUM_EPISODES):

    # ── Reset all vehicles and servers ───────────────────────
    vehicles[0].reset(x=0,  y=0)
    vehicles[1].reset(x=15, y=0)
    for server in servers.values():
        server.reset()

    pending_experiences = {}

    # ── Per-vehicle episode counters ─────────────────────────
    generated  = {v.vehicle_id: 0 for v in vehicles}
    dropped    = {v.vehicle_id: 0 for v in vehicles}
    completed  = {v.vehicle_id: 0 for v in vehicles}
    unfinished = {v.vehicle_id: 0 for v in vehicles}
    episode_reward = 0

    for step in range(NUM_STEPS):

        # 1. Move and tick all vehicles
        for v in vehicles:
            v.move()
            v.tick(step)

        # 2. Tick all servers
        for server in servers.values():
            server.transmission_tick()
            server.backhaul_tick()
            server.tick(step)

        # 3. Deliver completion rewards
        for server in servers.values():
            for task in server.completed_tasks:
                if task.task_id in pending_experiences:
                    exp = pending_experiences.pop(task.task_id)
                    final_reward = exp["immediate_reward"] + 1.0
                    agent.remember(
                        exp["state"], exp["action"],
                        final_reward, exp["next_state"]
                    )
                    agent.learn()
                    episode_reward += 1.0
                completed[task.owner_id] += 1
            server.completed_tasks.clear()

        for v in vehicles:
            for task in v.local_completed:
                if task.task_id in pending_experiences:
                    exp = pending_experiences.pop(task.task_id)
                    final_reward = exp["immediate_reward"] + 1.0
                    agent.remember(
                        exp["state"], exp["action"],
                        final_reward, exp["next_state"]
                    )
                    agent.learn()
                    episode_reward += 1.0
                completed[v.vehicle_id] += 1
            v.local_completed.clear()

        # 4. Process each vehicle in random order
        vehicle_order = random.sample(vehicles, len(vehicles))

        for v in vehicle_order:

            # Find active server for THIS vehicle
            active_server = None
            for server in servers.values():
                if server.in_range(v):
                    active_server = server
                    break

            # Handover detection
            if active_server is not None:
                if active_server.server_id != v.current_server_id:
                    if v.current_server_id is not None:
                        old_server    = servers[v.current_server_id]
                        waiting_tasks = old_server.release_queue()
                        num_tasks     = len(waiting_tasks)
                        for task in waiting_tasks:
                            shared_rate = (
                                old_server.backhaul_bandwidth
                                / max(num_tasks, 1)
                            )
                            migration_time = task.data_size / shared_rate
                            delay_steps = math.ceil(
                                migration_time / STEP_DURATION
                            )
                            active_server.accept_backhaul_task(
                                task, delay_steps
                            )
                    v.current_server_id = active_server.server_id

            # Generate and decide for each task
            if active_server is not None:
                tasks = v.generate_tasks(step)
                generated[v.vehicle_id] += len(tasks)

                for task in tasks:
                    # Contention indicator
                    other_on_server = sum(
                        1 for other in vehicles
                        if other.vehicle_id != v.vehicle_id
                        and other.current_server_id
                            == active_server.server_id
                    ) / 5.0

                    edge_load = (
                        len(active_server.task_queue) +
                        len(active_server.transmission_queue) +
                        len(active_server.backhaul_queue) +
                        (1 if active_server.current_task else 0)
                    )
                    local_busy = 1 if v.local_task else 0

                    state = [
                        edge_load / 50,
                        len(active_server.transmission_queue) / 20,
                        task.cpu_cycles / 1000,
                        task.data_size / 1_000_000,
                        len(v.local_queue) / 20,
                        local_busy,
                        other_on_server               # contention
                    ]

                    action = agent.act(state)

                    if action == 0:  # local
                        accepted = v.accept_local_task(task)
                        local_pressure = (
                            len(v.local_queue) +
                            (1 if v.local_task else 0)
                        ) / 5.0
                        contention_penalty = other_on_server * 0.3
                        immediate_reward = (
                            0.2 - local_pressure - contention_penalty
                        )

                    else:  # offload
                        distance = math.sqrt(
                            (v.x - active_server.x)**2 +
                            (v.y - active_server.y)**2
                        )
                        distance = max(1.0, distance)
                        delay_sec = channel.compute_delay(
                            task.data_size, distance
                        )
                        delay_steps = math.ceil(
                            delay_sec / STEP_DURATION
                        )
                        task.transmission_remaining = delay_steps
                        active_server.add_to_transmission_queue(task)

                        edge_pressure  = edge_load / 10.0
                        channel_cost   = delay_steps / 3.0
                        contention_penalty = other_on_server * 0.3
                        immediate_reward = (
                            0.2 - edge_pressure
                            - channel_cost
                            - contention_penalty
                        )

                    episode_reward += immediate_reward

                    next_state = [
                        (edge_load + 1) / 50,
                        len(active_server.transmission_queue) / 20,
                        task.cpu_cycles / 1000,
                        task.data_size / 1_000_000,
                        len(v.local_queue) / 20,
                        1 if v.local_task else 0,
                        other_on_server
                    ]

                    pending_experiences[task.task_id] = {
                        "state"           : state,
                        "action"          : action,
                        "immediate_reward": immediate_reward,
                        "next_state"      : next_state
                    }

            else:
                tasks = v.generate_tasks(step)
                generated[v.vehicle_id] += len(tasks)
                dropped[v.vehicle_id]   += len(tasks)
                v.current_server_id      = None

    # ── Episode end penalty — per vehicle ─────────────────────
    for v in vehicles:
        vid = v.vehicle_id
        unfinished[vid] = (
            len(v.local_queue) +
            (1 if v.local_task else 0)
        )

    for server in servers.values():
        for task in (list(server.task_queue) +
                     list(server.transmission_queue) +
                     list(server.backhaul_queue) +
                     ([server.current_task]
                      if server.current_task else [])):
            if task.owner_id in unfinished:
                unfinished[task.owner_id] += 1

    for vid, count in unfinished.items():
        penalty = -0.5 * count
        episode_reward += penalty

    pending_experiences.clear()

    # ── Record episode metrics ────────────────────────────────
    all_completed = sum(completed.values())
    all_generated = sum(generated.values())
    completion_rate = (
        all_completed / all_generated if all_generated > 0 else 0
    )

    rates = [
        completed[v.vehicle_id] / generated[v.vehicle_id]
        if generated[v.vehicle_id] > 0 else 0
        for v in vehicles
    ]
    n    = len(rates)
    jain = (
        (sum(rates)**2) / (n * sum(r**2 for r in rates))
        if any(rates) else 0
    )

    episode_metrics.append({
        "episode"        : episode,
        "completion_rate": completion_rate,
        "episode_reward" : episode_reward,
        "agent_epsilon"  : agent.epsilon,
        "jain_index"     : jain,
        "v1_completion"  : rates[0],
        "v2_completion"  : rates[1],
    })

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1:4d} | "
              f"reward={episode_reward:7.2f} | "
              f"completion={completion_rate:.1%} | "
              f"jain={jain:.3f} | "
              f"ε={agent.epsilon:.3f}")

# ── Save weights and metrics ──────────────────────────────────
agent.save("results/trained_agent_marl.npy")

with open("results/episode_metrics_marl.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=episode_metrics[0].keys())
    writer.writeheader()
    writer.writerows(episode_metrics)

print("\n=== Training Complete ===")
print(f"Episodes          : {NUM_EPISODES}")
print(f"Final epsilon     : {agent.epsilon:.4f}")
print(f"Memory size       : {len(agent.memory)}")
last = episode_metrics[-1]
print(f"Last reward       : {last['episode_reward']:.2f}")
print(f"Last completion   : {last['completion_rate']:.1%}")
print(f"Last Jain index   : {last['jain_index']:.3f}")
print(f"V1 completion     : {last['v1_completion']:.1%}")
print(f"V2 completion     : {last['v2_completion']:.1%}")
print(f"Weights saved     : results/trained_agent_marl.npy")
print(f"Metrics saved     : results/episode_metrics_marl.csv")