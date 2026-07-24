import csv
import matplotlib.pyplot as plt

episodes = []
rewards  = []

with open('results/episode_metrics_marl.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row['episode']))
        rewards.append(float(row['episode_reward']))

window = 50
rolling_avg = []
for i in range(len(rewards)):
    start = max(0, i - window + 1)
    avg   = sum(rewards[start:i+1]) / len(rewards[start:i+1])
    rolling_avg.append(avg)

plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards,
         color='lightgray', alpha=0.5, label='Episode Reward')
plt.plot(episodes, rolling_avg,
         color='blue', linewidth=2,
         label='50-Episode Moving Average')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('MARL DQN Learning Curve — Two Vehicles')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/learning_curve_marl.png', dpi=300)
plt.show()