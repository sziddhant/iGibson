from agent import ICMAgent
from runner import Runner
from utils import get_args

from math import pi
from gibson2.envs.igibson_env import iGibsonEnv
import gibson2
import numpy as np
import os

if __name__ == '__main__':

    """Argument parsing"""
    args = get_args()

    """Environment"""
    # create the atari environments
    # NOTE: this wrapper automatically resets each env if the episode is done
    env =iGibsonEnv(config_file=os.path.join(gibson2.example_config_path, 'fetch.yaml'),
                                  mode='gui',
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)
    state = env.reset()
    env.robots[0].set_position([0,0,0])
    env.robots[0].set_orientation([0,0,0,1])
    env.robots[0].keep_still()
    action = np.zeros(env.action_space.shape)

    # action[0],action[1]= vel_control(dist,theta)
    # state, reward, done, _ = nav_env.step(action)

    # env = make_atari_env(args.env_name, num_env=args.num_envs, seed=args.seed)
    # env = VecFrameStack(env, n_stack=args.n_stack)

    """Agent"""
    agent = ICMAgent(2, 1, 2, lr=args.lr)


    """Train"""
    runner = Runner(agent, env, args.num_envs, args.n_stack, args.rollout_size, args.num_updates,
                    args.max_grad_norm, args.value_coeff, args.entropy_coeff,
                    args.tensorboard, args.log_dir, args.cuda, args.seed)
    runner.train()