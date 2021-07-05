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

ang_k=4
lin_k=-5

def approx_angle(p1,p2):
    x=p1[0]-p2[0]
    y=p1[1]-p2[1]
    ans= math.atan2(y,x)
    print("ans=",ans,x,y)
    return ans

def vel_control(dist,theta):
    ang_vel = ang_k*theta
    lin_vel = lin_k*dist
    
    if abs(theta)>0.1:
        lin_vel=0
    print("Vel= ", lin_vel,ang_vel)
    wheel_axle_half = 0.2
    # wheel_radius = 1
    left_wheel_ang_vel = -(lin_vel - ang_vel *wheel_axle_half) 
    right_wheel_ang_vel = -(lin_vel + ang_vel *wheel_axle_half)
    return(left_wheel_ang_vel,right_wheel_ang_vel)

def run():
    nav_env = iGibsonEnv(config_file=os.path.join(gibson2.example_config_path, 'fetch.yaml'),
                                  mode='gui',
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)

    motion_planner = MotionPlanningWrapper(nav_env,base_mp_algo='birrt',)
    state = nav_env.reset()
    print (state)
    pass
    nav_env.robots[0].set_position([0,0,0])
    nav_env.robots[0].set_orientation([0,0,0,1])
    nav_env.robots[0].keep_still()

    way_points=motion_planner.plan_base_motion(np.array([-2.5,2.5,0]))
    if way_points is None:
        way_points=motion_planner.plan_base_motion(np.array([-3,3,0]))
        print(way_points)
    print(way_points)
    motion_planner.dry_run_base_plan(way_points)

    
    # nav_env.robots[0].
    # nav_env.task.target_pos= [-1,0.1,0]
    # nav_env.robots[0].control[:2] ="position"
    # print(nav_env.robots[0].wheel_radius)# ="position"
    # print(nav_env.robots[0].calc_state())
    # print(nav_env.robots[0].joint_velocity)
    # print(nav_env.robots[0].wheel_axle_half)
    # random_floor = nav_env.scene.get_random_floor()


    # p1 =np.array([0,0,0.1])
    # p2 = nav_env.scene.get_random_point(random_floor)[1]

    # shortest_path, geodesic_distance = nav_env.scene.get_shortest_path(
    #     random_floor, p1[:2], p2[:2], entire_path=True)
    # print('random point 1:', p1)
    # print('random point 2:', p2)
    # print('geodesic distance between p1 and p2', geodesic_distance)
    # print('shortest path from p1 to p2:', shortest_path)

    # print(nav_env.action_space.sample())
    # nav_env.task.load_visualization(nav_env)
    # print(nav_env.task.goal_format)
    # # print(nav_env.task.target_pos)
    # nav_env.task.target_pos=np.array([-2.5,2.5,0])
    y=nav_env.task.get_shortest_path(nav_env,entire_path=True)[0]
    x=nav_env.task.get_shortest_path(nav_env)
    print(x[0][-1])
    motion_plan=motion_planner.plan_base_motion([x[0][-1][0],x[0][-1][1],0])
    print(motion_plan)
    # x=x[0]
    # nav_env.robots[0].set_position([x[-1][0],x[-1][1],0])
    # x=nav_env.task.get_shortest_path(nav_env)
    # print(x)
    # x=x[0]
    # nav_env.robots[0].set_position([x[-1][0],x[-1][1],0])
    # x=nav_env.task.get_shortest_path(nav_env)
    # print(x)
    # x=x[0]
    # nav_env.robots[0].set_position([x[-1][0],x[-1][1],0])
    # print(nav_env.task.get_task_obs(nav_env))
    # motion_planner.dry_run_base_plan(x)


    i=1

    D=0.05
    while True:
        sub_goal=[motion_plan[i][0],motion_plan[i][1]]
        current_pos= nav_env.robots[0].get_position()[:2]
        print("Current pos =",current_pos)
        ang=approx_angle(sub_goal,current_pos)
        
        robot_angle=nav_env.robots[0].get_rpy()[2]
        theta= ang-robot_angle
        print("theta= ", theta)
        dist = math.sqrt((current_pos[0]-sub_goal[0])**2+(current_pos[1]-sub_goal[1])**2)
        # print(dist)
        if dist<D:
            if i==len(x[0])-1:
                x=nav_env.task.get_shortest_path(nav_env)
                motion_plan=motion_planner.plan_base_motion([x[0][-1][0],x[0][-1][1],0])
                i=1
            i+=1
        print(i)
        
        action = np.zeros(nav_env.action_space.shape)
        print(nav_env.action_space.shape)

        action[0],action[1]= vel_control(dist,theta)
        print(action[:2])
        # action[1]=1
        # action[2]=1
        # nav_env.task.step(nav_env)
        # action[:2]=x[i]
        # action[0]=motion_plan[i][1]
        state, reward, done, _ = nav_env.step(action)

if __name__ == "__main__":
    run()

