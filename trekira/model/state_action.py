from trekira.sim.quadruped import Unitree  # TODO: Make this generic and enforced later
from trekira.reward.simple import Gait
from trekira.model.action_reward import A_R_Simple

import torch as pt
import numpy as np

pt.autograd.set_detect_anomaly(True)

class S_A_Simple:
    def __init__(self, sim: Unitree, action_reward: A_R_Simple):
        self.sim = sim
        self.action_shape = [self.sim.getJointCount()]
        self.state_shape = [3 + 4 + 12]
        self.model = None
        self.optimizer = None 
        self.simReward = Gait(sim)
        self.action_reward = action_reward
        self.thresh = 0.1
        self.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    def compile(self):
        self.model = pt.nn.Sequential(
            pt.nn.Linear(self.state_shape[0], 64),
            pt.nn.ReLU(),
            pt.nn.Linear(64, 128),
            pt.nn.ReLU(),
            pt.nn.Linear(128, 64),
            pt.nn.ReLU(),
            pt.nn.Linear(64, 32),
            pt.nn.ReLU(),
            pt.nn.Linear(32, self.action_shape[0]),
            pt.nn.Tanh()
        )
        self.model.to(self.device)
        self.model.share_memory()
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=0.01)

    def train_step(self):
        reward = 0
        best_action = None
        current_state = None
        counter = 0
        print(f">>>Finding best action for {self.sim.name}")
        # while reward < self.thresh and counter < 100:
        for _ in range(100):
            self.optimizer.zero_grad()
            joints = [s[0] for s in self.sim.currentState.get_joints()]
            states = np.concatenate((self.sim.currentState.get_xyz(), self.sim.currentState.get_quant(), joints), axis=0)
            current_state = pt.tensor(states, dtype=pt.float32)
            action = self.model(pt.tensor(states, dtype=pt.float32).to(self.device))
            best_action = action
            reward = self.action_reward.model(action, pt.tensor(states, dtype=pt.float32).to(self.device))
            loss = 100 - reward#pt.div(1, reward)
            loss.backward()
            print(f"\rEst Reward: {reward.item()}, Loss: {loss.item()}", end="")
            self.optimizer.step() 
            counter += 1
        print("<<<Best action found")
        self.action_reward.run_sim(best_action, current_state)
        self.action_reward.train(5)
        print(f"!!!>>> Round Complete with {self.simReward.get_distance()} <<<!!!")

    def save(self, path):
        pt.save(self.model.state_dict(), path)
