from trekira.sim.quadruped import Unitree
from trekira.reward.simple import Gait
from trekira.model.action_reward import A_R_Simple
from trekira.model.state_action import S_A_Simple

import pybullet as p
import numpy as np
import torch as pt

unitree = Unitree()
p.setRealTimeSimulation(1)
reward = Gait(unitree)

model = A_R_Simple(unitree)
model.compile()

model2 = S_A_Simple(unitree, model)
model2.compile()

while (1):
    # jointPoseEnd = 4 * [unitree.reset_pos[0], unitree.reset_pos[1], -2]
    # poses = np.linspace(4 * unitree.reset_pos, jointPoseEnd, 1000)
    # poses_back = np.linspace(jointPoseEnd, 4 * unitree.reset_pos, 1000)
    # poses = np.concatenate((poses, poses_back), axis=0)
    # for pose in poses:
    #     loss = model2.train_step(pose)
    #     print(f"Loss: {loss}")
    loss = model2.train_step()
    rpy = unitree.currentState.get_rpy()
    print(unitree.check_feet_on_ground())
    if unitree.check_feet_on_ground():
        unitree.resetAll()
        unitree.reset()
        print("!!!!!!!!!!!!!!!!!!!!RESET!!!!!!!!!!!!!!!!!!!!")
    #
    # if p.readUserDebugParameter(unitree.resetButtonId) > 0.5:
    #     unitree.resetAll()
    #     unitree.reset()
    #     print("!!!!!!!!!!!!!!!!!!!!RESET!!!!!!!!!!!!!!!!!!!!")

