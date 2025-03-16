from trekira.sim.quadruped import Unitree
from trekira.reward.simple import Gait
from trekira.model.action_reward import A_R_Simple
from trekira.model.state_action import S_A_Simple

import pybullet as p
import numpy as np
import torch as pt

import torch.multiprocessing as mp

def train_model(model_index, epochs):
    for _ in range(epochs):
        sa_models[model_index].train_step()
        if unitrees[model_index].check_feet_on_ground():
            unitrees[model_index].resetAll()
            unitrees[model_index].reset()
            print("!!!!!!!!!!!!!!!!!!!!RESET!!!!!!!!!!!!!!!!!!!!")

INSTANCES = 10
unitrees = []

for i in range(INSTANCES):
    unitrees.append(Unitree(str(i)))

p.setRealTimeSimulation(1)

rewards = []
for unitree in unitrees:
    rewards.append(Gait(unitree))

ar_models = []
for unitree in unitrees:
    ar_models.append(A_R_Simple(unitree))
    ar_models[-1].compile()

sa_models = []
for unitree, model in zip(unitrees, ar_models):
    sa_models.append(S_A_Simple(unitree, model))
    sa_models[-1].compile()

# use mp
epochs = 50

if __name__ == '__main__':
    pt.multiprocessing.set_start_method('spawn')


    processes = []
    for i in range(INSTANCES):
        processes.append(mp.Process(target=train_model, args=(i, epochs)))

    for process in processes:
        process.start()

    are_all_done = False
    while not are_all_done:
        are_all_done = True
        for process in processes:
            are_all_done = are_all_done and not process.is_alive()

    for process in processes:
        process.join()

    for i in range(INSTANCES):
        print(f"Unitree {i} has reached {rewards[i].get_distance()} meters")
        ar_models[i].save(f"unitree_{i}_ar.pt")
        sa_models[i].save(f"unitree_{i}_sa.pt")


# while (1):
    # jointPoseEnd = 4 * [unitree.reset_pos[0], unitree.reset_pos[1], -2]
    # poses = np.linspace(4 * unitree.reset_pos, jointPoseEnd, 1000)
    # poses_back = np.linspace(jointPoseEnd, 4 * unitree.reset_pos, 1000)
    # poses = np.concatenate((poses, poses_back), axis=0)
    # for pose in poses:
    #     loss = model2.train_step(pose)
    #     print(f"Loss: {loss}")
    # loss = model2.train_step()
    # rpy = unitree.currentState.get_rpy()
    # if unitree.check_feet_on_ground():
        # unitree.resetAll()
        # unitree.reset()
        # print("!!!!!!!!!!!!!!!!!!!!RESET!!!!!!!!!!!!!!!!!!!!")
    #
    # if p.readUserDebugParameter(unitree.resetButtonId) > 0.5:
    #     unitree.resetAll()
    #     unitree.reset()
    #     print("!!!!!!!!!!!!!!!!!!!!RESET!!!!!!!!!!!!!!!!!!!!")

