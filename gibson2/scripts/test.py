from bokeh.events import PressUp
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import gibson2
import os
env = iGibsonEnv(config_file=os.path.join(gibson2.example_config_path, 'fetch.yaml'),
                                  mode='headless',
                                  action_timestep=1.0 / 10.0,
                                  physics_timestep=1.0 / 240.0)

motion_planner = MotionPlanningWrapper(env,base_mp_algo='birrt',)



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class NeuralNetwork(nn.Module):
    def __init__(self,n_inputs,n_outputs):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, n_outputs),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.flatten(x)
        # x = x.to(device)
        # try:
        #     x = torch.from_numpy(x).float() #.astype(np.float32)
        # except:
        #     pass
        logits = self.linear_stack(x)
        return logits

BATCH_SIZE = 1
BATCH_SIZE_NEW = 512
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# Get number of actions from gym action space
n_actions = 2

policy_net = NeuralNetwork(3,2).to(device)
policy_net.float()
target_net = NeuralNetwork(3,2).to(device)
target_net.float()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print(policy_net(state))
            ret = np.zeros(10)
            ret_temp = policy_net(state).numpy()
            ret[0] = ret_temp[0]
            ret[1] = ret_temp[1]
            # print(type(ret))
            return ret
    else:
        ret = np.zeros(10)
        ret_temp = np.random.rand(2)
        ret[0] = ret_temp[0]*2-1
        ret[1] = ret_temp[1]*2-1
        return ret #torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []



def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By definition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    SA_Val = []
    E_EA_Val = []
    for _ in range(BATCH_SIZE_NEW):
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        while batch.next_state[0] == None:
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # print(state_batch.shape)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch.float()) #.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((BATCH_SIZE,2), device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states.float()) #.max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values.float()
        E_EA_Val.append(expected_state_action_values)
        SA_Val.append(state_action_values)

    E_EA_Val = torch.cat(E_EA_Val).float()
    SA_Val = torch.cat(SA_Val).float()


    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    criterion.float()
    loss = criterion(state_action_values, expected_state_action_values.float())
    

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes, such as 300+ for meaningful
# duration improvements.
#
wp= env.task.get_shortest_path(env,entire_path=True)[0][1]
print(wp)
# env.task.target_pos= np.array([wp[0],wp[1],0])
num_episodes = 50
for i_episode in range(num_episodes):
    print(i_episode)
    # Initialize the environment and state
    env.reset()
    state = np.array([env.robots[0].get_position()[0],env.robots[0].get_position()[1],env.robots[0].get_rpy()[2]])
    state =  torch.from_numpy(state).float()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        st, reward, done, _ = env.step(action)
        print(st.keys())
        print(st['occupancy_grid'].shape)
        if t%50 ==0:
            plt.figure(1)
            plt.imshow(np.reshape(st['occupancy_grid'],(128,128)))
            # plt.imshow(st['occupancy_grid'])
            plt.show()

        # Observe new state
        if not done:
            next_state = np.array([env.robots[0].get_position()[0],env.robots[0].get_position()[1],env.robots[0].get_rpy()[2]])
            
            reward = -((state[0]-wp[0])**2+(state[0]-wp[0])**2)**0.5
            # print(i_episode,wp, reward,  next_state)
            if abs(state[0]-wp[0])<0.01 and abs(state[1]-wp[1])<0.01:
                done=True
                print(env.task.get_termination(env))
            
            next_state = torch.from_numpy(next_state).float()
            # print(type(next_state))
        else:
            next_state = None
        if done:
            next_state = None
            reward = 5
        # Store the transition in memory
        action = torch.from_numpy(action).float()
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state


        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
# env.render()
# env.close()
plt.ioff()
plt.show()


