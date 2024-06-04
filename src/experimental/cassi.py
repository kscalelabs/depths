"""
https://github.com/nmansard/jnrh2023/blob/bad0f9638721b09e8e134188e8fad96624fbaa08/1_invgeom.ipynb
https://app.element.io/#/room/%23jnrh2023-tuto%3Alaas.fr?via=laas.fr
"""
import casadi
import joblib
import numpy as np
import pinocchio as pin
from numpy.linalg import norm
from pinocchio import casadi as cpin

# Define the robot model
robot = pin.robot_wrapper.RobotWrapper.BuildFromURDF("data/humanoid.urdf")
model = robot.model
data = robot.data
q_norm = pin.neutral(model)
head_id = model.getFrameId("head")

cmodel = cpin.Model(model)
cdata = cmodel.createData()
cq = casadi.SX.sym("q", cmodel.nq, 1)


pin.forwardKinematics(model, data, q_norm)
cpin.forwardKinematics(cmodel, cdata, cq)
PATH = "data/demo_vid.pkl"
results = joblib.load(PATH)
exit()
body1_joints = results['outputs//_DEMO/vid/img/000180.jpg']["3d_joints"][0]  # Replace with actual keypoints


error_tool = casadi.Function(
    "etool",
    [cq],
    [
        cpin.log6(
            cdata.oMf[tool_id].inverse() * cpin.SE3(transform_target_to_world)
        ).vector
    ],
)

opti = casadi.Opti()
var_q = opti.variable(model.nq)
totalcost = casadi.sumsqr(error_tool(var_q))

opti.minimize(totalcost)
opti.solver("ipopt")  # select the backend solver

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_q = opti.value(var_q)
except:
    print("ERROR in convergence, plotting debug info.")
    sol_q = opti.debug.value(var_q)

print(
    "The robot finally reached effector placement at\n",
    robot.placement(sol_q, 6),
)