from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator
import numpy as np
from gibson2.utils.utils import parse_config
from gibson2.robots.fetch_robot import Fetch
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper

import os
import time
import gibson2


def main():


    s = Simulator(mode='gui', image_width=512,
                  image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False,
        load_object_categories=['chair','door','bed'])
    s.import_ig_scene(scene)

    config = parse_config(os.path.join(gibson2.example_config_path, 'fetch_reaching.yaml'))
    fetch = Fetch(config)
    s.import_robot(fetch)
    motion_planner=MotionPlanningWrapper(scene)

    home =np.array([0,0,0])
    goal =np.array([0,0,-0.00876713])
    # fetch.load()
    fetch.set_position(home)
    fetch.robot_specific_reset()
    fetch.keep_still()
    # print("DIM====",fetch.action_dim)



    np.random.seed(0)
    random_floor = scene.get_random_floor()
    p1 = home#scene.get_random_point(random_floor)[1]
    p2 = scene.get_random_point(random_floor)[1]
    shortest_path, geodesic_distance = scene.get_shortest_path(
        random_floor, p1[:2], p2[:2], entire_path=True)
    print('random point 1:', p1)
    print('random point 2:', p2)
    print('geodesic distance between p1 and p2', geodesic_distance)
    print('shortest path from p1 to p2:', shortest_path)
    
    for _ in range(100000):
        
        s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
