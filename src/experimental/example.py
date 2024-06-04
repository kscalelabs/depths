import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

robot = RobotWrapper.BuildFromURDF("data/humanoid.urdf")
model = robot.model
data = robot.data
q_norm = pin.neutral(model)
pin.forwardKinematics(model, data, q_norm)