from trekira.sim.quadruped import Unitree  # TODO: Make this generic and enforced later
from trekira.reward.simple import Gait

import torch as pt
import numpy as np
from tqdm import tqdm

class Net(pt.nn.Module):
    def __init__(self, sim: Unitree):
        super(Net, self).__init__()
        self.action_shape = [sim.getJointCount()]
        self.state_shape = [3 + 4 + 12]
        self.input_shape = [np.sum(self.state_shape + self.action_shape)]
        self.reward_shape = [1]

        self.fc1 = pt.nn.Linear(self.input_shape[0], 64)
        self.fc2 = pt.nn.Linear(64, 32)
        self.fc3 = pt.nn.Linear(32, self.reward_shape[0])

    def forward(self, x, states):
        x = pt.concatenate((states, x), axis=0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class A_R_Simple:
    def __init__(self, sim: Unitree):
        self.sim = sim
        self.action_shape = [self.sim.getJointCount()]
        self.state_shape = [3 + 4 + 12]
        self.input_shape = [np.sum(self.state_shape + self.action_shape)]
        self.reward_shape = [1]
        self.model = None
        self.optimizer = None 
        self.simReward = Gait(sim)

        self.all_inputs = []
        self.all_outputs = []
        self.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    def compile(self):
        self.model = Net(self.sim).to(self.device)
        self.model.share_memory()
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=0.0001)
        self.sim.step(4*self.sim.reset_pos)

    # def train_step(self, action):
    #     self.optimizer.zero_grad()
    #     joints = [s[0] for s in self.sim.currentState.get_joints()]
    #     states = np.concatenate((self.sim.currentState.get_xyz(), self.sim.currentState.get_quant(), joints), axis=0)
    #     input = np.concatenate((states, action), axis=0)
    #     input = pt.tensor(input, dtype=pt.float32)
    #     
    #     self.sim.step(action)
    #     sim_reward = pt.tensor(self.simReward())
    #
    #     reward = self.model(input)
    #     loss = pt.square(pt.subtract(reward, sim_reward))
    #     loss.backward()
    #     for name, param in self.model.named_parameters():
    #         print(f"Gradient of {name}: {param.grad}")
    #     self.optimizer.step()
    #     return (sim_reward, reward, loss) 
    #
    # def forward_step(self, action, states):
    #     self.optimizer.zero_grad()
    #     reward = self.model(action, states)
    #     
    #     self.sim.step(action)
    #     sim_reward = pt.tensor(self.simReward())
    #
    #     loss = pt.square(pt.subtract(reward, sim_reward))
    #     loss.backward(retain_graph=True)
    #     # self.optimizer.step()
    #     self.all_inputs.append((action, states))
    #     self.all_outputs.append(sim_reward)
    #     return (sim_reward, reward, loss) 

    def run_sim(self, action, states):
        self.sim.step(action.cpu().detach().numpy())
        reward = self.simReward()
        self.all_inputs.append((action.cpu().detach().numpy(), states.cpu().detach().numpy()))
        self.all_outputs.append(reward)
        print(f"Simulated Reward: {reward}")

    def train(self, epochs=5):
        print(">>>Training the Action-Reward model")
        for _ in range(epochs):
            for (input, output) in zip(self.all_inputs, self.all_outputs):
                self.optimizer.zero_grad()
                action, states = input
                reward = self.model(pt.tensor(action, dtype=pt.float32).to(self.device), pt.tensor(states, dtype=pt.float32).to(self.device))
                loss = pt.square(pt.subtract(reward, output))
                loss.backward()
                print(f"\r{loss.item()}", end="")
                self.optimizer.step()
        print("<<<Training complete")

    def save(self, path):
        pt.save(self.model.state_dict(), path)
