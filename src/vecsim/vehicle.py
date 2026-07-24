import math
import random

from vecsim.task import Task


class Vehicle:

    def __init__(self, vehicle_id, x, y, speed, direction):
        # Basic vehicle attributes
        self.vehicle_id = vehicle_id
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction

        # Task management
        self.task_counter = 0

        # Local processing attributes
        self.local_capacity = 150
        self.local_task = None
        self.local_queue = []
        self.local_completed = []
        self.local_tasks_processed = 0

        # Handover tracking
        self.current_server_id = None

    def reset(self, x, y):
        self.x = x
        self.y = y

        self.task_counter = 0

        # Reset local computing state
        self.local_task = None
        self.local_queue = []
        self.local_completed = []
        self.local_tasks_processed = 0

        # Reset server association
        self.current_server_id = None

    def move(self):
        self.x += self.speed * math.cos(math.radians(self.direction))
        self.y += self.speed * math.sin(math.radians(self.direction))


    def generate_tasks(self, current_step, arrival_max=3):
        """
        Generate computation tasks.

        Parameters:
            current_step : current simulation time step
            arrival_max  : maximum number of tasks generated per step

        Default arrival_max=3 keeps the original high-load behavior.
        """

        tasks = []

        # C3b: configurable arrival intensity
        num_tasks = random.randint(0, arrival_max)

        for _ in range(num_tasks):

            task = Task(
                task_id=f"{self.vehicle_id}-{self.task_counter}",
                owner_id=self.vehicle_id,
                cpu_cycles=random.randint(200, 600),
                deadline=20
            )

            task.created_at = current_step

            self.task_counter += 1
            tasks.append(task)

        return tasks


    def __repr__(self):
        return (
            f"Vehicle {self.vehicle_id} | "
            f"pos: ({self.x:.1f}, {self.y:.1f})"
        )


    def tick(self, current_step):

        if self.local_task is None:
            return

        self.local_task.remaining_cycles -= self.local_capacity

        if self.local_task.remaining_cycles <= 0:

            self.local_task.status = "local_completed"
            self.local_task.completed_at = current_step

            self.local_task.latency = (
                current_step -
                self.local_task.created_at
            )

            self.local_tasks_processed += 1

            self.local_completed.append(
                self.local_task
            )

            self.local_task = None

            # Start next queued task
            if self.local_queue:

                next_task = self.local_queue.pop(0)

                next_task.status = "local_processing"

                self.local_task = next_task


    def accept_local_task(self, task):

        if self.local_task is None:

            self.local_task = task
            task.status = "local_processing"

            return True

        else:

            self.local_queue.append(task)
            task.status = "local_queued"

            return False



if __name__ == "__main__":

    v = Vehicle(
        vehicle_id=1,
        x=0,
        y=0,
        speed=2,
        direction=45
    )

    print("Default high-load test:")
    for step in range(3):

        v.move()

        tasks = v.generate_tasks(
            current_step=step
        )

        print(
            f"Step {step}: {v} | generated {len(tasks)} tasks"
        )


    print("\nLow-load test (arrival_max=1):")

    v.reset(x=0, y=0)

    for step in range(3):

        tasks = v.generate_tasks(
            current_step=step,
            arrival_max=1
        )

        print(
            f"Step {step}: generated {len(tasks)} tasks"
        )