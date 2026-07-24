import csv
import matplotlib.pyplot as plt
import numpy as np

regimes = ['low', 'medium', 'high']
window  = 50

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, regime in enumerate(regimes):
    episodes = []
    v1_rates = []
    v2_rates = []
    jain_scores = []

    with open(f'results/episode_metrics_marl_{regime}.csv') as f:
        for row in csv.DictReader(f):
            episodes.append(int(row['episode']))
            v1_rates.append(float(row['v1_completion']))
            v2_rates.append(float(row['v2_completion']))
            jain_scores.append(float(row['jain_index']))

    # Rolling averages
    def rolling(data, w):
        return [
            sum(data[max(0, i-w+1):i+1]) / len(data[max(0, i-w+1):i+1])
            for i in range(len(data))
        ]

    v1_roll   = rolling(v1_rates,   window)
    v2_roll   = rolling(v2_rates,   window)
    jain_roll = rolling(jain_scores, window)

    ax  = axes[idx]
    ax2 = ax.twinx()

    ax.plot(episodes, v1_roll,
            color='steelblue', linewidth=2, label='V1 completion')
    ax.plot(episodes, v2_roll,
            color='darkorange', linewidth=2, label='V2 completion')
    ax2.plot(episodes, jain_roll,
             color='green', linewidth=1.5,
             linestyle='--', label="Jain index")

    ax.set_xlabel('Episode')
    ax.set_ylabel('Completion Rate')
    ax2.set_ylabel('Jain Index')
    ax2.set_ylim(0.5, 1.05)
    ax.set_title(f'{regime.capitalize()} Load')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='lower right', fontsize=8)

fig.suptitle(
    'Per-Vehicle Fairness Dynamics — Three Load Regimes',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results/fairness_dynamics.png', dpi=300)
plt.show()