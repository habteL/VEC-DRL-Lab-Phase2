import csv
import matplotlib.pyplot as plt
import numpy as np

regimes     = ['low', 'medium', 'high']
colors      = ['green', 'orange', 'red']
window      = 50

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Learning curves for all three regimes ──────────────
for regime, color in zip(regimes, colors):
    episodes = []
    rates    = []
    with open(f'results/episode_metrics_marl_{regime}.csv') as f:
        for row in csv.DictReader(f):
            episodes.append(int(row['episode']))
            rates.append(float(row['completion_rate']))

    rolling = []
    for i in range(len(rates)):
        start = max(0, i - window + 1)
        rolling.append(sum(rates[start:i+1]) / len(rates[start:i+1]))

    axes[0].plot(episodes, rolling,
                 color=color, linewidth=2, label=f'{regime} load')

axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Completion Rate')
axes[0].set_title('MARL Learning Curves — Three Load Regimes')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ── Right: Final completion rate bar chart ───────────────────
final_rates = []
for regime in regimes:
    with open(f'results/episode_metrics_marl_{regime}.csv') as f:
        rows = list(csv.DictReader(f))
        last_50 = [float(r['completion_rate']) for r in rows[-50:]]
        final_rates.append(np.mean(last_50))

bars = axes[1].bar(regimes, final_rates,
                   color=colors, alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Load Regime')
axes[1].set_ylabel('Avg Completion Rate (last 50 episodes)')
axes[1].set_title('Final Performance by Load Regime')
axes[1].set_ylim(0, 1.0)
axes[1].grid(alpha=0.3, axis='y')

for bar, rate in zip(bars, final_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{rate:.1%}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('results/regime_comparison.png', dpi=300)
plt.show()