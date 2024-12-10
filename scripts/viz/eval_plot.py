import numpy as np
import os
import sys
stamp = "010607"

# file_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '/'
# eval = np.load(f'{file_prefix}eval_metrics.npz')
# suc = eval['success_list']
# timo = eval['timeout_list']
# ast = eval['avg_success_time']
# col = eval['collision_list']
# eval_acts = np.load(f'{file_prefix}eval_actions.npz')
# for key in eval_acts.keys():
#     print(key)
#     print(len(eval_acts[key]))
step = 2
import torch
t2_prefix = os.environ['HOME'] + '/imitation_learning_ros/src/imitation_learning/logs/' + str(stamp) + '_2/'
traj = f'{t2_prefix}eval_trajectories_{step}.pth'
trajectories = torch.load(traj)
print(trajectories)