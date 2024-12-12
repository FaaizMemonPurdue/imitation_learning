import numpy as np
import matplotlib.pyplot as plt

# Data
data = {
    'Successes': np.array([0, 3, 16, 15, 23, 26, 27, 27, 27, 30]),
    'Timeouts': np.array([30, 27, 14, 15, 7, 4, 3, 3, 3, 0]),
    'Collisions': np.array([824, 632, 561, 521, 208, 172, 180, 125, 93, 95]),
    'Average Success Time': np.array([np.nan, 71, 47, 45, 43, 37, 36.5, 33, 32.7, 35])
}

data = {
    'Successes': np.array([0, 0, 0, 7, 10, 13, 17, 20, 23, 25]),
    'Timeouts': np.array([30, 30, 30, 23, 20, 17, 13, 10, 7, 5]),
    'Collisions': np.array([791, 801, 650, 626, 544, 462, 422, 297, 215, 133]),
    'Average Success Time': np.array([np.nan, np.nan, np.nan, 97, 92, 86, 80, 81, 79, 63])
}
data = {
    'Successes': np.array([0, 0, 2, 0, 9, 15, 21, 18, 24, 27]),
    'Timeouts': np.array([30, 30, 28, 30, 21, 15, 9, 12, 6, 3]),
    'Collisions': np.array([931, 780, 742, 751, 683, 595, 521, 487, 402, 358]),
    'Average Success Time': np.array([np.nan, np.nan, 95, np.nan, 78, 69, 57, 61, 53, 48])
}
import numpy as np
import os
import sys
# stamp = "062739" #yawaware
# stamp = "010607" #2IWIL
stamp = "191804" #lstick
t2_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '_2/'
eval = np.load(f'{t2_prefix}eval_metrics.npz')
suc = eval['success_list']
timo = eval['timeout_list']
ast = eval['avg_success_time']
col = eval['collision_list']
data = {
    'Successes': suc,
    'Timeouts': timo,
    'Average Success Time': ast,
    'Collisions': col,
}
x = np.arange(len(data['Successes']))

fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot Successes and Timeouts on the first y-axis
ax1.plot(x, data['Successes'], 'o-', label='Successes', color='blue')
ax1.plot(x, data['Timeouts'], 'o-', label='Timeouts', color='orange')
ax1.set_xlabel('Evaluation Index', fontsize=14)
ax1.set_ylabel('Successes / Timeouts', color='blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue', labelsize=18)
ax1.tick_params(axis='x', labelsize=12)

# Create a second y-axis for Collisions
ax2 = ax1.twinx()
ax2.plot(x, data['Collisions'], 'o-', label='Collisions', color='green')
ax2.set_ylabel('Collisions', color='green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='green', labelsize=18)

# Create a third y-axis for Average Success Time
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60)) # Offset the third axis
ax3.plot(x, data['Average Success Time'], 'o-', label='Average Success Time', color='red')
ax3.set_ylabel('Average Success Time', color='red', fontsize=14)
ax3.tick_params(axis='y', labelcolor='red', labelsize=18)

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, fontsize=12)

plt.title('Strafe (state-constrained) 2IWIL-TRPO', fontsize=16)
plt.xticks(x)
plt.grid(True)
plt.tight_layout()
plt.show()
