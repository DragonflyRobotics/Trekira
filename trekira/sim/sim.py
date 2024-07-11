from abc import ABC, abstractmethod

class Simulation(ABC):
    def __init__(self):
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def getJointCount(self):
        pass
