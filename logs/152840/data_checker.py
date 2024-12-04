import torch
import numpy as np

collect_actions = torch.load('logs/152840/collect_actions.pt')
eval_actions = np.load('logs/152840/eval_actions.npz')
eval_metrics = np.load('logs/152840/eval_metrics.npz')

print("A")