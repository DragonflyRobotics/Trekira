from trekira.sim.quadruped import Unitree
from trekira.model.action_reward import A_R_Simple
from trekira.model.state_action import S_A_Simple


import pybullet as p
import torch as pt

INSTANCES = 10
CHECK_NUMBER = 0

ar_models_str = []
for i in range(INSTANCES):
    ar_models_str.append(f"unitree_{i}_ar.pt")

sr_models_str = []
for i in range(INSTANCES):
    sr_models_str.append(f"unitree_{i}_sa.pt")


unitree = Unitree("Main")

p.setRealTimeSimulation(1)


ar_model = A_R_Simple(unitree)
ar_model.compile()
ar_model.model.load_state_dict(pt.load(ar_models_str.pop(0)))

sa_model = S_A_Simple(unitree, ar_model)
sa_model.compile()
sa_model.model.load_state_dict(pt.load(sr_models_str.pop(0)))



while True:
    sa_model.train_step()
    if unitree.check_feet_on_ground():
        unitree.resetAll()
        unitree.reset()
        print("!!!!!!!!!!!!!!!!!!!!RESET!!!!!!!!!!!!!!!!!!!!")
