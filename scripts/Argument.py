from argparse import ArgumentParser


def arguments():

    parser = ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env', type=str, default="Reacher-v1", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=1111, metavar='N',
                        help='random seed (default: 1111')
    parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                        help='size of a single batch')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--fname', type=str, default='expert', metavar='F',
                        help='the file name to save trajectory')
    parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train an expert')
    parser.add_argument('--hidden-dim', type=int, default=100, metavar='H',
                        help='the size of hidden layers')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                        help='learning rate')
    parser.add_argument('--weight', action='store_true',
                        help='consider confidence into loss')
    parser.add_argument('--only', action='store_true',
                        help='only use labeled samples')
    parser.add_argument('--noconf', action='store_true',
                        help='use only labeled data but without conf')
    parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                        help='number of iterations of value function optimization iterations per each policy optimization step')
    parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                        help='learning rate of value network')
    parser.add_argument('--noise', type=float, default=0.0, metavar='N')
    parser.add_argument('--loss-type', type=str, default='attentioncu', choices=['cu', 'attentioncu', 'confU'], metavar='L')
    parser.add_argument('--eval-epochs', type=int, default=3, metavar='E',
                        help='epochs to evaluate model')
    parser.add_argument('--prior', type=float, default=0.2,
                        help='ratio of confidence data')
    parser.add_argument('--initialization', type=str, default="orthogonal")
    parser.add_argument('--traj-size', type=int, default=2000)
    parser.add_argument('--ofolder', type=str, default='log')
    parser.add_argument('--ifolder', type=str, default='demonstrations')
    args = parser.parse_args()

    return args