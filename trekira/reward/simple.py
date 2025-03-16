from trekira.sim.quadruped import Unitree  # TODO: Make this generic and enforced later
import pybullet as p
import numpy as np

class Gait:
    def __init__(self, sim: Unitree):
        self.sim = sim

    def compute(self):
        distance = self.sim.currentState.get_xyz()[1]/10.0
        rpy = self.sim.currentState.get_rpy()
        (roll, pitch, yaw) = np.degrees(rpy)/70.0
        velocities = p.getBaseVelocity(self.sim.model)
        (vx, vy, vz) = velocities[0]
        (wx, wy, wz) = velocities[1]
        speed_penalty = np.sqrt(vx**2 + vy**2 + vz**2)/10.0 
        angular_penalty = np.sqrt(wx**2 + wy**2 + wz**2)/10.0
        angle_penalty = np.sqrt(roll**2 + pitch**2 + yaw**2)
        return (2 * distance) - speed_penalty - angular_penalty - angle_penalty

    def get_distance(self):
        return self.sim.currentState.get_xyz()[1]

    def get_loss(self):
        return 1/self.compute()

    def __call__(self):
        return self.compute()
