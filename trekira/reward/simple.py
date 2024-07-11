from trekira.sim.quadruped import Unitree  # TODO: Make this generic and enforced later

class Gait:
    def __init__(self, sim: Unitree):
        self.sim = sim

    def compute(self):
        (roll, pitch, yaw) = self.sim.currentState.get_rpy()
        return self.sim.currentState.get_xyz()[1] + (1/roll) + (1/pitch) + (1/yaw)

    def get_loss(self):
        return 1/self.compute()

    def __call__(self):
        return self.compute()
