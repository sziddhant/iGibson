#####  Keyboard control


from math import pi
from operator import sub
from networkx.generators.classic import wheel_graph
from numpy.lib.function_base import angle
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import gibson2
import argparse
import numpy as np
import os
import math
import pybullet as p
from gibson2.render.profiler import Profiler
import csv

act=[]
def run():
    nav_env = iGibsonEnv(config_file=os.path.join(gibson2.example_config_path, 'fetch.yaml'),
                                  mode='gui',
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)

    motion_planner = MotionPlanningWrapper(nav_env,base_mp_algo='birrt',)
    state = nav_env.reset()
    nav_env.robots[0].set_position([0,0,0])
    nav_env.robots[0].set_orientation([0,0,0,1])
    nav_env.robots[0].keep_still()

    Turn_check=False
    a1=0
    a2=0
    act=[]
    states=[]
    end= False
    while not end:
        with Profiler("Simulation step"):
            keys = p.getKeyboardEvents()
            print(keys)
        for k, v in keys.items():
            if k == 65297:
                if Turn_check:
                    a1=0
                    a2=0
                    Turn_check=False
                a1+= 0.1
                a2+=0.1
            elif k == 65298:
                if Turn_check:
                    a1=0
                    a2=0
                    Turn_check=False
                a1-= 0.1
                a2-=0.1
            elif k == 65295:
                a1= 0.2
                a2= -0.2
                Turn_check=True
            elif k == 65296:
                a1= -0.2
                a2= 0.2
                Turn_check=True
            elif k == 32:
                end=True


        action = np.zeros(nav_env.action_space.shape)
        action[0]=a1
        action[1]=a2
        save=[action[0],action[1],nav_env.robots[0].get_position()[0],nav_env.robots[0].get_position()[1],nav_env.robots[0].get_rpy()[2]]
        states.append(save)
        act.append(action[:2])
        print(action[:2])
        state, reward, done, _ = nav_env.step(action)
        # print(state,action,reward)       
    act=np.array(act) 
    states= np.array(states)
    np.savetxt("action.csv", act, delimiter=",")
    np.savetxt("Save.csv", states, delimiter=",")
if __name__ == "__main__":
    run()

