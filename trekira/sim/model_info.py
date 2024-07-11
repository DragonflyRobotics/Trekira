import numpy as np
import pybullet as p

class PhysicalInfo:
    def __init__(self, xyz, quant, joints):
        self.xyz = xyz
        self.quant = quant
        self.rpy = np.degrees(p.getEulerFromQuaternion(quant)) 
        self.pose = [self.xyz, self.rpy]
        self.joints = joints

    def __str__(self):
        return 'Physical Info | xyz: ' + str(self.xyz) + ' rpy: ' + str(self.rpy) + ' quant: ' + str(self.quant) + ' joints: ' + str(self.joints)

    def get_pose(self):
        return tuple(self.pose)

    def get_joints(self):
        return tuple(self.joints)

    def get_quant(self):
        return tuple(self.quant)

    def get_xyz(self):
        return tuple(self.xyz)

    def get_rpy(self):
        return tuple(self.rpy)
