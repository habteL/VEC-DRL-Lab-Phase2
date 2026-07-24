import csv
import random
import math
import argparse
import numpy as np

from vecsim.vehicle import Vehicle
from vecsim.edge_server import EdgeServer
from vecsim.channel import WirelessChannel
from vecsim.agent import DQNAgent


# ── Command line configuration ────────────────────────────────
parser = argparse.ArgumentParser()

parser.add_argument(
    "--arrival-max",
    type=int,
    default=3,
    help="Maximum tasks generated per vehicle per step"
)

parser.add_argument(
    "--regime",
    type=str,
    default="high",
    help="Load regime name"
)

args = parser.parse_args()

ARRIVAL_MAX = args.arrival_max
REGIME = args.regime


random.seed(42)
np.random.seed(42)


STEP_DURATION = 0.1
NUM_STEPS = 35
NUM_EPISODES = 1000


# ── Vehicles ─────────────────────────────────────────────────
vehicles = [
    Vehicle(vehicle_id=1, x=0, y=0, speed=2, direction=0),
    Vehicle(vehicle_id=2, x=15, y=0, speed=1, direction=0),
]


# ── Servers ──────────────────────────────────────────────────
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


agent = DQNAgent(
    state_size=7,
    action_size=2
)


episode_metrics = []
step_metrics = []

pending_experiences = {}


print("=== MARL Training ===")
print(f"Regime       : {REGIME}")
print(f"Arrival max  : {ARRIVAL_MAX}")
print(f"Episodes     : {NUM_EPISODES}")
print()


for episode in range(NUM_EPISODES):

    vehicles[0].reset(x=0, y=0)
    vehicles[1].reset(x=15, y=0)

    for server in servers.values():
        server.reset()


    pending_experiences = {}

    episode_latencies = []


    generated = {
        v.vehicle_id: 0
        for v in vehicles
    }

    completed = {
        v.vehicle_id: 0
        for v in vehicles
    }


    unfinished = {
        v.vehicle_id: 0
        for v in vehicles
    }


    episode_reward = 0


    for step in range(NUM_STEPS):


        # ── Vehicle update ───────────────────────────────────
        for v in vehicles:
            v.move()
            v.tick(step)


        # ── Server update ────────────────────────────────────
        for server in servers.values():

            server.transmission_tick()
            server.backhaul_tick()
            server.tick(step)



        # ── Completion rewards ───────────────────────────────
        for server in servers.values():

            for task in server.completed_tasks:


                if hasattr(task, "latency"):

                    episode_latencies.append(
                        task.latency
                    )


                if task.task_id in pending_experiences:

                    exp = pending_experiences.pop(
                        task.task_id
                    )


                    final_reward = (
                        exp["immediate_reward"]
                        + 1.0
                    )


                    agent.remember(
                        exp["state"],
                        exp["action"],
                        final_reward,
                        exp["next_state"]
                    )

                    agent.learn()


                completed[task.owner_id] += 1


            server.completed_tasks.clear()



        for v in vehicles:

            for task in v.local_completed:


                if hasattr(task, "latency"):

                    episode_latencies.append(
                        task.latency
                    )


                if task.task_id in pending_experiences:

                    exp = pending_experiences.pop(
                        task.task_id
                    )


                    final_reward = (
                        exp["immediate_reward"]
                        + 1.0
                    )


                    agent.remember(
                        exp["state"],
                        exp["action"],
                        final_reward,
                        exp["next_state"]
                    )


                    agent.learn()


                completed[v.vehicle_id] += 1


            v.local_completed.clear()



        # ── Random vehicle order ─────────────────────────────
        vehicle_order = random.sample(
            vehicles,
            len(vehicles)
        )


        for v in vehicle_order:


            active_server = None


            for server in servers.values():

                if server.in_range(v):

                    active_server = server
                    break



            if active_server is not None:


                if active_server.server_id != v.current_server_id:


                    if v.current_server_id is not None:


                        old_server = servers[
                            v.current_server_id
                        ]


                        waiting_tasks = (
                            old_server.release_queue()
                        )


                        num_tasks = len(waiting_tasks)


                        for task in waiting_tasks:


                            shared_rate = (
                                old_server.backhaul_bandwidth /
                                max(num_tasks,1)
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



                tasks = v.generate_tasks(
                    step,
                    arrival_max=ARRIVAL_MAX
                )


                generated[v.vehicle_id] += len(tasks)



                for task in tasks:


                    other_on_server = sum(

                        1 for other in vehicles

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

                        (1 if active_server.current_task else 0)

                    )



                    state = [

                        edge_load / 50,

                        len(active_server.transmission_queue)/20,

                        task.cpu_cycles / 1000,

                        task.data_size / 1_000_000,

                        len(v.local_queue)/20,

                        1 if v.local_task else 0,

                        other_on_server

                    ]


                    action = agent.act(state)



                    if action == 0:


                        v.accept_local_task(task)

                        reward = 0.2



                    else:


                        distance = max(

                            1.0,

                            math.sqrt(

                                (v.x-active_server.x)**2

                                +

                                (v.y-active_server.y)**2

                            )

                        )


                        delay_sec = channel.compute_delay(

                            task.data_size,

                            distance

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


                        reward = 0.2



                    episode_reward += reward


                    next_state = state.copy()


                    pending_experiences[
                        task.task_id
                    ] = {

                        "state":state,

                        "action":action,

                        "immediate_reward":reward,

                        "next_state":next_state

                    }



            else:


                tasks = v.generate_tasks(
                    step,
                    arrival_max=ARRIVAL_MAX
                )


                generated[v.vehicle_id] += len(tasks)



        # ── Queue evolution record ────────────────────────────

        step_metrics.append({

            "episode":episode,

            "step":step,

            "total_queue":

                sum(

                    len(s.task_queue)

                    for s in servers.values()

                ),

            "avg_latency":

                np.mean(episode_latencies)
                if episode_latencies
                else 0

        })



    # ── Metrics ──────────────────────────────────────────────

    total_completed = sum(completed.values())

    total_generated = sum(generated.values())


    completion_rate = (

        total_completed /

        total_generated

        if total_generated > 0

        else 0

    )


    episode_metrics.append({

        "episode":episode,

        "completion_rate":completion_rate,

        "reward":episode_reward,

        "avg_latency":

            np.mean(episode_latencies)
            if episode_latencies
            else 0

    })


    if (episode+1)%100==0:

        print(
            f"Episode {episode+1} | "
            f"Completion={completion_rate:.2%}"
        )



agent.save(
    f"results/trained_agent_marl_{REGIME}.npy"
)


with open(
    f"results/episode_metrics_marl_{REGIME}.csv",
    "w",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=episode_metrics[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        episode_metrics
    )


with open(
    f"results/step_metrics_marl_{REGIME}.csv",
    "w",
    newline=""
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=step_metrics[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        step_metrics
    )


print("\n=== Training Complete ===")
print(f"Regime saved: {REGIME}")