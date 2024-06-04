"""
1. We start from random q - which are actions!
2. We compute forward dynamics with pinnochio and get poses in general frame
3. The casadi takes this poses and computes against motion cap poses for keypoints
4. The objective is to minimize the difference between robot keypoints and MoCap keypoints
5. How do we get dynamics ?? We can get dynamics from pinnochio??
6. We iterate with casadi where obj function is given by pinnochio
"""
from collections import OrderedDict

import casadi as ca
import joblib
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Load the MoCap data
PATH = "data/demo_vid.pkl"
results = joblib.load(PATH)

# Define the robot model
robot = RobotWrapper.BuildFromURDF("assets/humanoid.urdf")
model = robot.model
data = robot.data

JOINTS = 45
num_frames = len(results)
indices = [ii for ii in range(0, 45)]

keypoints = OrderedDict({
    "head": 18,  # good
    "neck": 40,  # good
    "shoulder_right": 2,
    "shoulder_left": 34,
    "elbow_right": 3,
    "elbow_left": 6,
    "wrist_right": 31,  # good
    "wrist_left": 36, # good
    "torso": 27,
    "knee_right":  26,  # good
    "knee_left":  29,  # good
    "foot_right":  21, # good
    "foot_left":  25,   # good
})
indices = [keypoints[key] for key in keypoints]

# indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# Extract 3D joint data
mocap_data = []
for key, item in results.items():
    mocap_data.append(item["3d_joints"][0][indices])


mocap_data = np.array(mocap_data) # T, K, 3
# Number of frames and keypoints
N = 1 # mocap_data.shape[0] - only one frame - let's see if we can adjust
n_keypoints = len(keypoints)


print("Poses on Y axis which are related to the height!")
print(mocap_data[0][[0, 8, 12]])
print("Height of the robot is", mocap_data[0][12][1] - mocap_data[0][0][1])
mocap_to_robot = OrderedDict({
    "neck": "neck",
    "knee_right": "right_knee"
})

# Define optimization variables
q = ca.MX.sym('q', 2) #model.nq)
qd = ca.MX.sym('qd', 2) # model.nv)


# Function to compute the positions of the keypoints on the robot
def robot_keypoints(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    keypoints = []

    for _, joint  in mocap_to_robot.items():
        frame_id = model.getFrameId(joint)  # Assuming keypoints are named 'keypoint_0', 'keypoint_1', etc.
        keypoints.append(data.oMf[frame_id].translation)
    return ca.vertcat(*keypoints)

# Define the objective function and constraints
objective = 0
constraints = []
dt = 0.01  # Time step
q_traj = [ca.MX.sym(f'q_{t}', model.nq) for t in range(N)]
qd_traj = [ca.MX.sym(f'qd_{t}', model.nv) for t in range(N)]
q_traj = [ca.MX.sym(f'q_{t}', 2) for t in range(N)]
qd_traj = [ca.MX.sym(f'qd_{t}', 2) for t in range(N)]

for t in range(N):
    q_t = q_traj[t]
    mocap_t = mocap_data[t].reshape(-1)
    robot_kp_t = robot_keypoints(q_t).reshape(-1)
    breakpoint()
    # Objective: minimize the difference between robot keypoints and MoCap keypoints
    objective += ca.sumsqr(robot_kp_t - mocap_t)
    
    if t > 0:
        q_prev = q_traj[t-1]
        qd_prev = qd_traj[t-1]
        qd_t = qd_traj[t]
        
        # Constraint: q[t+1] = q[t] + (qd[t] + qd[t-1]) / 2 * dt
        # constraints.append(q_t - (q_prev + (qd_t + qd_prev) / 2 * dt))
        constraints.append(q_t)

# Create the NLP solver
nlp = {
    'x': ca.vertcat(*q_traj, *qd_traj),
    'f': objective,
    'g': ca.vertcat(*constraints)
}

opts = {
    'ipopt.print_level': 0,
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6
}

solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the optimization problem
sol = solver(x0=np.zeros((model.nq + model.nv) * N), lbg=0, ubg=0)

# Extract the solution
q_sol = ca.reshape(sol['x'][:model.nq * N], (N, model.nq))
qd_sol = ca.reshape(sol['x'][model.nq * N:], (N, model.nv))

# Print the results
print('Optimized joint positions (q):', q_sol)
print('Optimized joint velocities (qd):', qd_sol)