"""


# Step 3: Add the Conda Forge channel
conda config --add channels conda-forge
conda config --set channel_priority strict

# Step 4: Install Pinocchio
conda install pinocchio
conda install casadi

python PHALP/scripts/demo_phalp.py video.source=PHALP/assets/videos/vid.mp4
"""
import os
from os.path import abspath, dirname, join
from sys import argv

import numpy as np

# breakpoint()
import pinocchio
import pinocchio as pin
from numpy.linalg import norm, solve
from pinocchio.robot_wrapper import RobotWrapper

# model = pinocchio.buildSampleModelManipulator()
# data  = model.createData()

# JOINT_ID = 6
# oMdes = pinocchio.SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))

# q      = pinocchio.neutral(model)
# eps    = 1e-4
# IT_MAX = 1000
# DT     = 1e-1
# damp   = 1e-12


# i=0
# while True:
#     pinocchio.forwardKinematics(model, data, q)
#     iMd = data.oMi[JOINT_ID].actInv(oMdes)
#     err = pinocchio.log(iMd).vector  # in joint frame
#     if norm(err) < eps:
#         success = True
#         break
#     if i >= IT_MAX:
#         success = False
#         break
#     J = pinocchio.computeJointJacobian(model,data,q,JOINT_ID)  # in joint frame
#     J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
#     v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
#     q = pinocchio.integrate(model,q,v*DT)
#     if not i % 10:
#         print('%d: error = %s' % (i, err.T))
#     i += 1

# if success:
#     print("Convergence achieved!")
# else:
#     print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

# print('\nresult: %s' % q.flatten().tolist())
# print('\nfinal error: %s' % err.T)

 
# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

# You should change here to set up your own URDF file or just pass it as an argument of this example.
# urdf_filename = "stompy/robot.urdf"
# model = pinocchio.buildModelFromUrdf(urdf_filename)

xml_filename = "data/humanoid.urdf"
model = pinocchio.buildModelFromUrdf(xml_filename)

print('model name: ' + model.name)
 
# Create data required by the algorithms
data = model.createData()

# Initialize the configuration vector q with zeros (neutral position)
q = pinocchio.neutral(model)
# Initialize the configuration vector q with the neutral configuration

# Update the data structure to reflect the new configuration
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# Print the default keypoints (joint positions)
print("Default keypoints (joint positions):")
for joint_id, joint in enumerate(model.joints):
    if joint_id == 0:
        continue  # Skip the universe joint
    # joint_placement = pinocchio.SE3.Identity()
    joint_placement = data.oMi[joint_id]
    print(f"Joint {joint_id}")
    print(f" - Translation: {joint_placement.translation}")
    print(f" - Rotation: \n{joint_placement.rotation}")

# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print('q: %s' % q.T)

for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat )))

# moving the body with the respect to q sequentially
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
 
# Print out the placement of each joint of the kinematic tree
breakpoint()
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}".format( name, *oMi.translation.T.flat )))

pos = q

# from casadi import *
# # Symbols/expressions
# x = MX.sym('x')
# y = MX.sym('y')
# z = MX.sym('z')
# f = x**2+100*z**2
# g = z+(1-x)**2-y

# nlp = {}                 # NLP declaration
# nlp['x']= vertcat(x,y,z) # decision vars
# nlp['f'] = f             # objective
# nlp['g'] = g             # constraints

# # Create solver instance
# F = nlpsol('F','ipopt',nlp);

# # Solve the problem using a guess
# F(x0=[2.5,3.0,0.75],ubg=0,lbg=0)