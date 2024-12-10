env = "Ant-v2"
gamma = 0.995
tau = 0.97
l2_reg = 0.001
max_kl = 0.01
damping = 0.1
seed = 1111
batch_size = 100
log_interval = 1
fname = 'expert'
num_epochs = 500
hidden_dim = 100
lr = 0.001
weight = True
only = False
noconf = False
vf_iters = 30
vf_lr = 0.0003
noise = 0.0
loss_type = 'attentioncu'
eval_epochs = 3
prior = 0.2
initialization = 'orthogonal'
traj_size = 1445
ofolder = 'log2out'
ifolder = 'demonstrations'
save_interval = 1