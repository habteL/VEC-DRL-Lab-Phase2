import random
import math
import numpy as np
import matplotlib.pyplot as plt

from vecsim.vehicle     import Vehicle
from vecsim.edge_server import EdgeServer
from vecsim.channel     import WirelessChannel
from vecsim.agent       import DQNAgent


random.seed(123)
np.random.seed(123)


STEP_DURATION = 0.1
NUM_STEPS     = 35
NUM_EPISODES  = 100


def run_policy(policy_fn, num_episodes=NUM_EPISODES):

    # ── Two heterogeneous vehicles ───────────────────────────
    vehicles = [
        Vehicle(vehicle_id=1, x=0,  y=0, speed=2, direction=0),
        Vehicle(vehicle_id=2, x=15, y=0, speed=1, direction=0),
    ]

    # ── Three edge servers ───────────────────────────────────
    servers = {
        1: EdgeServer(
            server_id=1,
            capacity=300,
            x=10,
            y=0,
            coverage_radius=10
        ),
        2: EdgeServer(
            server_id=2,
            capacity=300,
            x=30,
            y=0,
            coverage_radius=10
        ),
        3: EdgeServer(
            server_id=3,
            capacity=300,
            x=50,
            y=0,
            coverage_radius=10
        ),
    }

    channel = WirelessChannel(
        bandwidth=5_000_000,
        transmission_power=1000,
        noise_factor=1
    )

    all_completion_rates = []
    all_jain_scores      = []


    for episode in range(num_episodes):

        # ── Reset environment ────────────────────────────────
        vehicles[0].reset(x=0, y=0)
        vehicles[1].reset(x=15, y=0)

        for server in servers.values():
            server.reset()


        generated = {
            v.vehicle_id: 0 for v in vehicles
        }

        completed = {
            v.vehicle_id: 0 for v in vehicles
        }

        dropped = {
            v.vehicle_id: 0 for v in vehicles
        }


        # ── Simulation steps ─────────────────────────────────
        for step in range(NUM_STEPS):

            # Move vehicles
            for v in vehicles:
                v.move()
                v.tick(step)


            # Update servers
            for server in servers.values():
                server.transmission_tick()
                server.backhaul_tick()
                server.tick(step)


            # Collect completed tasks
            for server in servers.values():
                for task in server.completed_tasks:
                    completed[task.owner_id] += 1

                server.completed_tasks.clear()


            for v in vehicles:
                for task in v.local_completed:
                    completed[v.vehicle_id] += 1

                v.local_completed.clear()


            # Random vehicle processing order
            vehicle_order = random.sample(
                vehicles,
                len(vehicles)
            )


            for v in vehicle_order:

                # Find connected server
                active_server = None

                for server in servers.values():
                    if server.in_range(v):
                        active_server = server
                        break


                # ── Handover handling ────────────────────────
                if active_server is not None:

                    if active_server.server_id != v.current_server_id:

                        if v.current_server_id is not None:

                            old_server = servers[v.current_server_id]

                            waiting_tasks = (
                                old_server.release_queue()
                            )

                            num_tasks = len(waiting_tasks)

                            for task in waiting_tasks:

                                shared_rate = (
                                    old_server.backhaul_bandwidth /
                                    max(num_tasks, 1)
                                )

                                migration_time = (
                                    task.data_size /
                                    shared_rate
                                )

                                delay_steps = math.ceil(
                                    migration_time /
                                    STEP_DURATION
                                )

                                active_server.accept_backhaul_task(
                                    task,
                                    delay_steps
                                )


                        v.current_server_id = (
                            active_server.server_id
                        )


                # ── Generate tasks ───────────────────────────
                if active_server is not None:

                    tasks = v.generate_tasks(step)

                    generated[v.vehicle_id] += len(tasks)


                    for task in tasks:


                        # Contention indicator
                        other_on_server = sum(
                            1
                            for other in vehicles
                            if other.vehicle_id != v.vehicle_id
                            and other.current_server_id ==
                            active_server.server_id
                        ) / 5.0


                        edge_load = (
                            len(active_server.task_queue)
                            +
                            len(active_server.transmission_queue)
                            +
                            len(active_server.backhaul_queue)
                            +
                            (1 if active_server.current_task
                             else 0)
                        )


                        # 7-dimensional state
                        state = [
                            edge_load / 50,
                            len(active_server.transmission_queue)
                            / 20,
                            task.cpu_cycles / 1000,
                            task.data_size / 1_000_000,
                            len(v.local_queue) / 20,
                            1 if v.local_task else 0,
                            other_on_server
                        ]


                        # Policy decision
                        action = policy_fn(state)


                        # ── Local execution ────────────────
                        if action == 0:

                            v.accept_local_task(task)


                        # ── Edge offloading ────────────────
                        else:

                            distance = math.sqrt(
                                (v.x - active_server.x) ** 2
                                +
                                (v.y - active_server.y) ** 2
                            )

                            distance = max(distance, 1.0)


                            delay_sec = (
                                channel.compute_delay(
                                    task.data_size,
                                    distance
                                )
                            )

                            delay_steps = math.ceil(
                                delay_sec /
                                STEP_DURATION
                            )

                            task.transmission_remaining = (
                                delay_steps
                            )

                            active_server.add_to_transmission_queue(
                                task
                            )


                # No available server
                else:

                    tasks = v.generate_tasks(step)

                    generated[v.vehicle_id] += len(tasks)

                    dropped[v.vehicle_id] += len(tasks)

                    v.current_server_id = None



        # ── Episode metrics ──────────────────────────────────

        rates = [
            completed[v.vehicle_id] /
            generated[v.vehicle_id]
            if generated[v.vehicle_id] > 0
            else 0
            for v in vehicles
        ]


        # Jain fairness index
        n = len(rates)

        jain = (
            (sum(rates) ** 2) /
            (n * sum(r ** 2 for r in rates))
            if any(rates)
            else 0
        )


        all_completion_rates.append(
            sum(rates) / n
        )

        all_jain_scores.append(
            jain
        )


    return {
        "completion_rate":
            float(np.mean(all_completion_rates)),

        "jain_index":
            float(np.mean(all_jain_scores))
    }



# =============================================================
# Policies
# =============================================================

def always_offload(state):
    return 1



def always_local(state):
    return 0



# Load trained MARL agent
trained_agent = DQNAgent(
    state_size=7,
    action_size=2
)

trained_agent.load(
    "results/trained_agent_marl.npy"
)

trained_agent.epsilon = 0.0



def marl_policy(state):

    return trained_agent.act(state)



# =============================================================
# Run comparison
# =============================================================

print("Running always-offload...")
offload_results = run_policy(always_offload)


print("Running always-local...")
local_results = run_policy(always_local)


print("Running trained MARL DQN...")
dqn_results = run_policy(marl_policy)



# =============================================================
# Print results
# =============================================================

print("\n=== MARL Baseline Comparison ===")

print(
    f"{'Policy':<20}"
    f"{'Completion Rate':>16}"
    f"{'Jain Index':>12}"
)

print("-" * 50)


print(
    f"{'Always Offload':<20}"
    f"{offload_results['completion_rate']:>15.1%}"
    f"{offload_results['jain_index']:>12.3f}"
)


print(
    f"{'Always Local':<20}"
    f"{local_results['completion_rate']:>15.1%}"
    f"{local_results['jain_index']:>12.3f}"
)


print(
    f"{'MARL DQN Agent':<20}"
    f"{dqn_results['completion_rate']:>15.1%}"
    f"{dqn_results['jain_index']:>12.3f}"
)



# =============================================================
# Plot comparison
# =============================================================

policies = [
    "Always Offload",
    "Always Local",
    "MARL DQN"
]


completion_rates = [
    offload_results["completion_rate"],
    local_results["completion_rate"],
    dqn_results["completion_rate"]
]


jain_scores = [
    offload_results["jain_index"],
    local_results["jain_index"],
    dqn_results["jain_index"]
]


x = np.arange(len(policies))

width = 0.35


fig, ax = plt.subplots(figsize=(10, 6))


ax.bar(
    x - width / 2,
    completion_rates,
    width,
    label="Completion Rate"
)


ax.bar(
    x + width / 2,
    jain_scores,
    width,
    label="Jain Index"
)


ax.set_xticks(x)

ax.set_xticklabels(
    policies,
    rotation=15
)

ax.set_ylabel("Value")

ax.set_ylim(0, 1.1)

ax.set_title(
    "MARL Policy Comparison — Two Vehicles"
)

ax.legend()

ax.grid(
    alpha=0.3
)


plt.tight_layout()

plt.savefig(
    "results/comparison_marl.png",
    dpi=300
)

plt.show()