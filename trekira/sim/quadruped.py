from trekira.sim import Simulation
from trekira.sim import PhysicalInfo

import pybullet as p
import time
import numpy as np

class Unitree(Simulation):
    def __init__(self, name):
        self.name = name
        self.TIMESTEP = 1./500.
        self.model = None
        self.jointIds = []
        self.paramIds = []
        self.jointOffsets = []
        self.jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        self.jointAngles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.maxForceId = None
        self.reset_pos = [0.04, 0.7, -1.4]

        self.previousState = None
        self.currentState = None

        self.initialBase = None

        self.resetButtonId = None
        self.force = 50

        super().__init__()
    
    def getJointCount(self):
        return 12

    def publish_params(self):
        index = 0
        for j in range(p.getNumJoints(self.model)):
            p.changeDynamics(self.model, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.model, j)
            js = p.getJointState(self.model, j)
            print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, (js[0]-self.jointOffsets[index])/self.jointDirections[index]))
                index = index + 1

    def reset(self):
        for _ in range(10):
            for j in range(self.getJointCount()):
                targetPos = float(self.reset_pos[j % 3])
                p.setJointMotorControl2(self.model, self.jointIds[j], p.POSITION_CONTROL, self.jointDirections[j]*targetPos + self.jointOffsets[j], force=self.force)
                p.stepSimulation()
                time.sleep(self.TIMESTEP)

    def resetAll(self):
        p.resetBasePositionAndOrientation(self.model, self.initialBase[0], self.initialBase[1])
        self.reset()

    def setup(self):
        p.connect(p.GUI)
        plane = p.loadURDF("plane/plane.urdf")
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.TIMESTEP)
        # p.setDefaultContactERP(0)
        # urdfFlags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
        urdfFlags = p.URDF_USE_SELF_COLLISION
        self.model = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.5], [0, 0.5, 0.5, 0], flags=urdfFlags, useFixedBase=False)
        lower_legs = [2, 5, 8, 11]
        for l0 in lower_legs:
            for l1 in lower_legs:
                if (l1 > l0):
                    enableCollision = 1
                    print("collision for pair", l0, l1, p.getJointInfo(self.model, l0)[self.getJointCount()], p.getJointInfo(self.model, l1)[self.getJointCount()], "enabled = ", enableCollision)
                    p.setCollisionFilterPair(self.model, self.model, l0, l1, enableCollision)


        for i in range(4):
            self.jointOffsets.append(0)
            self.jointOffsets.append(-0.7)
            self.jointOffsets.append(0.7)

        self.maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 70)
        self.resetButtonId = p.addUserDebugParameter("reset", 0, 1, 0)

        for j in range(p.getNumJoints(self.model)):
            p.changeDynamics(self.model, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.model, j)
            # print(info)
            # jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)


        p.getCameraImage(480, 320)
        p.setRealTimeSimulation(0)

        self.reset()
        self.publish_params()
        self.previousState = self.currentState
        self.currentState = p.getBasePositionAndOrientation(self.model)

        self.initialBase = p.getBasePositionAndOrientation(self.model)

        p.setRealTimeSimulation(1)

    def step(self, *args, **kwargs):
        states = args[0]
        for j in range(self.getJointCount()):
            targetPos = float(states[j])
            p.setJointMotorControl2(self.model, self.jointIds[j], p.POSITION_CONTROL, self.jointDirections[j]*targetPos + self.jointOffsets[j], force=self.force)
        p.stepSimulation()
        time.sleep(self.TIMESTEP)
        self.previousState = self.currentState
        self.currentState = PhysicalInfo(p.getBasePositionAndOrientation(self.model)[0], p.getBasePositionAndOrientation(self.model)[1], p.getJointStates(self.model, range(self.getJointCount())))
        return self.currentState

    def run(self):
        p.setRealTimeSimulation(1)
        while (1):
            jointPoseEnd = [self.reset_pos[0], self.reset_pos[1], -2]
            poses = np.linspace(self.reset_pos, jointPoseEnd, 1000)
            poses_back = np.linspace(jointPoseEnd, self.reset_pos, 1000)
            poses = np.concatenate((poses, poses_back), axis=0)
            for pose in poses:
                for j in range(self.getJointCount()):
                    targetPos = float(pose[j % 3])
                    p.setJointMotorControl2(self.model, self.jointIds[j], p.POSITION_CONTROL, self.jointDirections[j]*targetPos + self.jointOffsets[j], force=self.force)
                p.stepSimulation()
                time.sleep(self.TIMESTEP)
    
    def is_fallen(self):
        # Get the orientation of the robot's base
        pos, orn = p.getBasePositionAndOrientation(self.model)
        euler = p.getEulerFromQuaternion(orn)
        
        # Define fall criteria based on orientation and height
        max_tilt_angle = np.radians(70)  # Max tilt angle in radians (45 degrees)
        
        # Check if the robot is fallen based on tilt and height
        if abs(euler[0]) > max_tilt_angle: 
            return True
        return False

    def check_feet_on_ground(self):
        lower_legs = [2, 5, 8, 11]
        feet_on_ground = []
        for foot in lower_legs:
            contact_points = p.getContactPoints(bodyA=self.model, linkIndexA=foot)
            if contact_points:
                feet_on_ground.append(True)
            else:
                feet_on_ground.append(False)
        return not any(feet_on_ground) or self.is_fallen()
