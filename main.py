import pybullet as p
import time
import numpy as np

TIMESTEP = 1./500.

p.connect(p.GUI)
plane = p.loadURDF("plane/plane.urdf")
p.setGravity(0, 0, -9.8)
p.setTimeStep(TIMESTEP)
# p.setDefaultContactERP(0)
# urdfFlags = p.URDF_USE_SELF_COLLISION + p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS 
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, 0.5], [0, 0.5, 0.5, 0], flags=urdfFlags, useFixedBase=False)

# enable collision between lower legs

for j in range(p.getNumJoints(quadruped)):
    print(p.getJointInfo(quadruped, j))

# 2,5,8 and 11 are the lower legs
lower_legs = [2, 5, 8, 11]
for l0 in lower_legs:
    for l1 in lower_legs:
        if (l1 > l0):
            enableCollision = 1
            print("collision for pair", l0, l1, p.getJointInfo(quadruped, l0)[12], p.getJointInfo(quadruped, l1)[12], "enabled = ", enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, l0, l1, enableCollision)

jointIds = []
paramIds = []
jointOffsets = []
jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
jointAngles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(4):
    jointOffsets.append(0)
    jointOffsets.append(-0.7)
    jointOffsets.append(0.7)

maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 70)

for j in range(p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped, j)
    # print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        jointIds.append(j)


p.getCameraImage(480, 320)
p.setRealTimeSimulation(0)

joints = []
reset_pos = [0.04, 0.7, -1.4]
for _ in range(10):
    for j in range(12):
        targetPos = float(reset_pos[j % 3])
        p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, jointDirections[j]*targetPos + jointOffsets[j], force=p.readUserDebugParameter(maxForceId))
        p.stepSimulation()
        time.sleep(TIMESTEP)

index = 0
for j in range(p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped, j)
    js = p.getJointState(quadruped, j)
    print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, (js[0]-jointOffsets[index])/jointDirections[index]))
        index = index + 1


p.setRealTimeSimulation(1)

while (1):
    jointPoseEnd = [reset_pos[0], reset_pos[1], -2]
    poses = np.linspace(reset_pos, jointPoseEnd, 1000)
    poses_back = np.linspace(jointPoseEnd, reset_pos, 1000)
    poses = np.concatenate((poses, poses_back), axis=0)
    for pose in poses:
        for j in range(12):
            targetPos = float(pose[j % 3])
            p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, jointDirections[j]*targetPos + jointOffsets[j], force=p.readUserDebugParameter(maxForceId))
        p.stepSimulation()
        # print robot roll pitch yaw x y z 
        print(p.getBasePositionAndOrientation(quadruped), p.getEulerFromQuaternion(p.getBasePositionAndOrientation(quadruped)[1]))
        time.sleep(TIMESTEP)

