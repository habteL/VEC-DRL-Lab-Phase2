import math


class EdgeServer:
    def __init__(self, server_id, capacity, x=0, y=0, coverage_radius=10, backhaul_bandwidth=10_000_000):
        self.server_id = server_id
        self.capacity = capacity
        self.current_task = None
        self.task_queue = []          # waiting room
        self.tasks_processed = 0
        self.x = x  # Server's x-coordinate
        self.y = y  # Server's y-coordinate
        self.coverage_radius = coverage_radius  # Server's coverage radius
        # self.completed_tasks = []  # stores finished tasks for metrics
        self.transmission_queue = []  # tasks currently being transmitted
        self.completed_tasks = []   # per-step completion buffer
        self.backhaul_bandwidth = backhaul_bandwidth
        self.backhaul_queue = []  # tasks currently being transmitted over backhaul
    def reset(self):
        self.current_task = None
        self.task_queue = []
        self.tasks_processed = 0
        self.completed_tasks = []
        self.transmission_queue = []
        self.backhaul_queue = []
        self.completed_tasks = []
    def accept_task(self, task):
        if self.current_task is None:
            self.current_task = task
            task.status = "processing"
            return True
        else:
            self.task_queue.append(task)  # join the queue
            task.status = "queued"
            return False

    def tick(self, current_step):
        # Pull from queue if server is idle
        if self.current_task is None and self.task_queue:
            next_task = self.task_queue.pop(0)
            next_task.status = "processing"
            self.current_task = next_task

        if self.current_task is None:
            return
        self.current_task.remaining_cycles -= self.capacity

        if self.current_task.remaining_cycles <= 0:
            self.current_task.status = "completed"
            self.current_task.completed_at = current_step
            self.current_task.latency = (current_step
                                        - self.current_task.created_at)
            self.tasks_processed += 1
            self.completed_tasks.append(self.current_task)
            
            self.current_task = None

            if self.task_queue:
                next_task = self.task_queue.pop(0)
                next_task.status = "processing"
                self.current_task = next_task

    def __repr__(self):
        busy = "busy" if self.current_task else "idle"
        return (f"EdgeServer {self.server_id} | "
                f"capacity: {self.capacity} | "
                f"status: {busy} | "
                f"queue: {len(self.task_queue)} waiting | "
                f"processed: {self.tasks_processed} | "
                f"position: ({self.x}, {self.y}) | "
                f"backhaul queue: {len(self.backhaul_queue)} | "
                f"coverage radius: {self.coverage_radius}")
    def in_range(self, vehicle):
        # Placeholder for range checking logic
        distance = math.sqrt(((self.x - vehicle.x) ** 2 + (self.y - vehicle.y) ** 2))
        return distance <= self.coverage_radius
    def release_queue(self):
        tempo_task_list = self.task_queue.copy()
        self.task_queue.clear()
        return tempo_task_list
    def add_to_transmission_queue(self, task):
        task.status = "transmitting"
        task.transmission_delay = task.transmission_remaining
        self.transmission_queue.append(task)

    def transmission_tick(self):
        still_transmitting = []
        for task in self.transmission_queue:
            task.transmission_remaining -= 1
            if task.transmission_remaining <= 0:
               
                task.status = "queued"
                self.task_queue.append(task)
            else:
                still_transmitting.append(task)
        self.transmission_queue = still_transmitting
    def accept_backhaul_task(self, task, delay_steps):
        task.backhaul_remaining = delay_steps
        task.status = "backhaul_transit"
        self.backhaul_queue.append(task)

    def backhaul_tick(self):
        still_in_transit = []
        for task in self.backhaul_queue:
            task.backhaul_remaining -= 1
            if task.backhaul_remaining <= 0:
                task.status = "queued"
                self.task_queue.append(task)
            else:
                still_in_transit.append(task)
        self.backhaul_queue = still_in_transit