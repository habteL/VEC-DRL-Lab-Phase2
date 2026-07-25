import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from vecsim.vehicle     import Vehicle
from vecsim.edge_server import EdgeServer
from vecsim.channel     import WirelessChannel
from vecsim.agent       import DQNAgent

STEP_DURATION = 0.1
NUM_STEPS     = 35
NUM_EPISODES  = 100

def run_three_vehicles(agent, num_episodes=NUM_EPISODES, seed=123):
    random.seed(seed)
    np.random.seed(seed)

    vehicles = [
        Vehicle(vehicle_id=1, x=0,  y=0, speed=2,   direction=0),
        Vehicle(vehicle_id=2, x=15, y=0, speed=1,   direction=0),
        Vehicle(vehicle_id=3, x=5,  y=0, speed=1.5, direction=0),
    ]

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

    all_completion_rates = []
    all_jain_scores      = []
    per_vehicle_rates    = {v.vehicle_id: [] for v in vehicles}

    for episode in range(num_episodes):

        vehicles[0].reset(x=0,  y=0)
        vehicles[1].reset(x=15, y=0)
        vehicles[2].reset(x=5,  y=0)
        for server in servers.values():
            server.reset()

        generated = {v.vehicle_id: 0 for v in vehicles}
        completed = {v.vehicle_id: 0 for v in vehicles}
        dropped   = {v.vehicle_id: 0 for v in vehicles}

        for step in range(NUM_STEPS):

            for v in vehicles:
                v.move()
                v.tick(step)

            for server in servers.values():
                server.transmission_tick()
                server.backhaul_tick()
                server.tick(step)

            for server in servers.values():
                for task in server.completed_tasks:
                    completed[task.owner_id] += 1
                server.completed_tasks.clear()

            for v in vehicles:
                for task in v.local_completed:
                    completed[v.vehicle_id] += 1
                v.local_completed.clear()

            vehicle_order = random.sample(vehicles, len(vehicles))

            for v in vehicle_order:
                active_server = None
                for server in servers.values():
                    if server.in_range(v):
                        active_server = server
                        break

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

                if active_server is not None:
                    tasks = v.generate_tasks(step, arrival_max=3)
                    generated[v.vehicle_id] += len(tasks)

                    for task in tasks:
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

                        state = [
                            edge_load / 50,
                            len(active_server.transmission_queue) / 20,
                            task.cpu_cycles / 1000,
                            task.data_size / 1_000_000,
                            len(v.local_queue) / 20,
                            1 if v.local_task else 0,
                            other_on_server
                        ]

                        action = agent.act(state)

                        if action == 0:
                            v.accept_local_task(task)
                        else:
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

                else:
                    tasks = v.generate_tasks(step, arrival_max=3)
                    generated[v.vehicle_id] += len(tasks)
                    dropped[v.vehicle_id]   += len(tasks)
                    v.current_server_id      = None

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

        all_completion_rates.append(sum(rates) / n)
        all_jain_scores.append(jain)
        for v in vehicles:
            per_vehicle_rates[v.vehicle_id].append(
                completed[v.vehicle_id] / generated[v.vehicle_id]
                if generated[v.vehicle_id] > 0 else 0
            )

    return {
        'completion_rate': float(np.mean(all_completion_rates)),
        'jain_index'     : float(np.mean(all_jain_scores)),
        'per_vehicle'    : {
            vid: float(np.mean(rates))
            for vid, rates in per_vehicle_rates.items()
        }
    }


# ── Experiment 1: Zero-shot ───────────────────────────────────
print("Experiment 1: Zero-shot (2-vehicle agent on 3 vehicles)")
zero_shot_agent = DQNAgent(state_size=7, action_size=2)
zero_shot_agent.load('results/trained_agent_marl_high.npy')
zero_shot_agent.epsilon = 0.0
zero_shot_results = run_three_vehicles(zero_shot_agent)

print(f"  Completion rate : {zero_shot_results['completion_rate']:.1%}")
print(f"  Jain index      : {zero_shot_results['jain_index']:.3f}")
for vid, rate in zero_shot_results['per_vehicle'].items():
    print(f"  V{vid} completion : {rate:.1%}")

# ── Experiment 2: Retrain with 3 vehicles ────────────────────
print("\nExperiment 2: Retraining with 3 vehicles (1000 episodes)")
retrained_agent = DQNAgent(state_size=7, action_size=2)

# Training loop
random.seed(42)
np.random.seed(42)

vehicles_train = [
    Vehicle(vehicle_id=1, x=0,  y=0, speed=2,   direction=0),
    Vehicle(vehicle_id=2, x=15, y=0, speed=1,   direction=0),
    Vehicle(vehicle_id=3, x=5,  y=0, speed=1.5, direction=0),
]
servers_train = {
    1: EdgeServer(server_id=1, capacity=300, x=10, y=0, coverage_radius=10),
    2: EdgeServer(server_id=2, capacity=300, x=30, y=0, coverage_radius=10),
    3: EdgeServer(server_id=3, capacity=300, x=50, y=0, coverage_radius=10),
}
channel_train = WirelessChannel(
    bandwidth=5_000_000,
    transmission_power=1000,
    noise_factor=1
)

for episode in range(1000):
    vehicles_train[0].reset(x=0,  y=0)
    vehicles_train[1].reset(x=15, y=0)
    vehicles_train[2].reset(x=5,  y=0)
    for server in servers_train.values():
        server.reset()

    generated      = {v.vehicle_id: 0 for v in vehicles_train}
    completed      = {v.vehicle_id: 0 for v in vehicles_train}
    dropped        = {v.vehicle_id: 0 for v in vehicles_train}
    pending_exp    = {}
    episode_reward = 0

    for step in range(NUM_STEPS):

        for v in vehicles_train:
            v.move()
            v.tick(step)

        for server in servers_train.values():
            server.transmission_tick()
            server.backhaul_tick()
            server.tick(step)

        for server in servers_train.values():
            for task in server.completed_tasks:
                if task.task_id in pending_exp:
                    exp = pending_exp.pop(task.task_id)
                    retrained_agent.remember(
                        exp["state"], exp["action"],
                        exp["immediate_reward"] + 1.0,
                        exp["next_state"]
                    )
                    retrained_agent.learn()
                completed[task.owner_id] += 1
            server.completed_tasks.clear()

        for v in vehicles_train:
            for task in v.local_completed:
                if task.task_id in pending_exp:
                    exp = pending_exp.pop(task.task_id)
                    retrained_agent.remember(
                        exp["state"], exp["action"],
                        exp["immediate_reward"] + 1.0,
                        exp["next_state"]
                    )
                    retrained_agent.learn()
                completed[v.vehicle_id] += 1
            v.local_completed.clear()

        vehicle_order = random.sample(vehicles_train, len(vehicles_train))

        for v in vehicle_order:
            active_server = None
            for server in servers_train.values():
                if server.in_range(v):
                    active_server = server
                    break

            if active_server is not None:
                if active_server.server_id != v.current_server_id:
                    if v.current_server_id is not None:
                        old_server    = servers_train[v.current_server_id]
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

            if active_server is not None:
                tasks = v.generate_tasks(step, arrival_max=3)
                generated[v.vehicle_id] += len(tasks)

                for task in tasks:
                    other_on_server = sum(
                        1 for other in vehicles_train
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

                    state = [
                        edge_load / 50,
                        len(active_server.transmission_queue) / 20,
                        task.cpu_cycles / 1000,
                        task.data_size / 1_000_000,
                        len(v.local_queue) / 20,
                        1 if v.local_task else 0,
                        other_on_server
                    ]

                    action = retrained_agent.act(state)

                    if action == 0:
                        accepted = v.accept_local_task(task)
                        local_pressure = (
                            len(v.local_queue) +
                            (1 if v.local_task else 0)
                        ) / 5.0
                        contention_penalty = other_on_server * 0.3
                        immediate_reward   = (
                            0.2 - local_pressure - contention_penalty
                        )
                    else:
                        distance = math.sqrt(
                            (v.x - active_server.x)**2 +
                            (v.y - active_server.y)**2
                        )
                        distance = max(1.0, distance)
                        delay_sec = channel_train.compute_delay(
                            task.data_size, distance
                        )
                        delay_steps = math.ceil(
                            delay_sec / STEP_DURATION
                        )
                        task.transmission_remaining = delay_steps
                        active_server.add_to_transmission_queue(task)
                        edge_pressure      = edge_load / 10.0
                        channel_cost       = delay_steps / 3.0
                        contention_penalty = other_on_server * 0.3
                        immediate_reward   = (
                            0.2 - edge_pressure
                            - channel_cost - contention_penalty
                        )

                    episode_reward += immediate_reward
                    pending_exp[task.task_id] = {
                        "state"           : state,
                        "action"          : action,
                        "immediate_reward": immediate_reward,
                        "next_state"      : state
                    }

            else:
                tasks = v.generate_tasks(step, arrival_max=3)
                generated[v.vehicle_id] += len(tasks)
                dropped[v.vehicle_id]   += len(tasks)
                v.current_server_id      = None

    pending_exp.clear()

    if (episode + 1) % 200 == 0:
        rates = [
            completed[v.vehicle_id] / generated[v.vehicle_id]
            if generated[v.vehicle_id] > 0 else 0
            for v in vehicles_train
        ]
        print(f"  Episode {episode+1:4d} | "
              f"completion={sum(rates)/len(rates):.1%} | "
              f"ε={retrained_agent.epsilon:.3f}")

retrained_agent.save('results/trained_agent_3vehicles.npy')

# ── Evaluate retrained agent ──────────────────────────────────
retrained_agent.epsilon = 0.0
retrained_results = run_three_vehicles(retrained_agent)

print(f"\nRetrained results:")
print(f"  Completion rate : {retrained_results['completion_rate']:.1%}")
print(f"  Jain index      : {retrained_results['jain_index']:.3f}")
for vid, rate in retrained_results['per_vehicle'].items():
    print(f"  V{vid} completion : {rate:.1%}")

# ── Summary table ─────────────────────────────────────────────
print("\n=== C3d Scalability Summary ===")
print(f"{'Config':<25} {'Completion':>12} {'Jain':>8}")
print("-" * 48)
print(f"{'2 vehicles (trained)':<25} {'46.0%':>12} {'0.992':>8}")
print(f"{'3 vehicles (zero-shot)':<25} "
      f"{zero_shot_results['completion_rate']:>11.1%} "
      f"{zero_shot_results['jain_index']:>7.3f}")
print(f"{'3 vehicles (retrained)':<25} "
      f"{retrained_results['completion_rate']:>11.1%} "
      f"{retrained_results['jain_index']:>7.3f}")

# ── Bar chart ─────────────────────────────────────────────────
configs     = ['2V\nTrained', '3V\nZero-shot', '3V\nRetrained']
completions = [
    0.460,
    zero_shot_results['completion_rate'],
    retrained_results['completion_rate']
]
jains = [
    0.992,
    zero_shot_results['jain_index'],
    retrained_results['jain_index']
]

x     = np.arange(len(configs))
width = 0.35
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.bar(x - width/2, completions, width,
        label='Completion Rate', color='steelblue', alpha=0.85)
ax2.bar(x + width/2, jains, width,
        label='Jain Index', color='darkorange', alpha=0.85)

ax1.set_ylabel('Completion Rate')
ax2.set_ylabel('Jain Fairness Index')
ax1.set_ylim(0, 0.8)
ax2.set_ylim(0.5, 1.1)
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.set_title('C3d Scalability: 2 vs 3 Vehicles')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('results/scalability_c3d.png', dpi=300)
plt.show()