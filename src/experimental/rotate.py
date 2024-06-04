from collections import OrderedDict

# Example keypoints for two bodies
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

NUM_FRAMES = 150


def extract_key_joints(joints):
    # Example indices for key joints (you may need to adjust based on your specific joint mapping)
    hip_index = 27
    shoulder_index = 34
    head_index = 18
    
    hip = joints[hip_index]
    shoulder = joints[shoulder_index]
    head = joints[head_index]
    
    return hip, shoulder, head


def compute_orientation(hip, shoulder, head):
    # Compute the vectors along the body
    torso_vector = shoulder - hip
    head_vector = head - shoulder
    
    # Normalize vectors
    torso_vector /= np.linalg.norm(torso_vector)
    head_vector /= np.linalg.norm(head_vector)
    
    # Compute the orientation matrix (assuming Y-axis as up direction)
    z_axis = torso_vector
    y_axis = np.cross(head_vector, torso_vector)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    return rotation_matrix


def apply_transformation(joints, transformation_matrix):
    rotated = np.dot(joints, transformation_matrix.T)
    return rotated


def load_model_get_keypoints(path="data/humanoid.urdf"):
    # Load the URDF file
    urdf_filename = path
    model = pin.buildModelFromUrdf(urdf_filename)
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    keypoints = []
    for _, oMi in zip(model.names, data.oMi):
        keypoints.append(oMi.translation.T.flat[:])

    return model, data, np.array(keypoints)


# Visualization
def visualize_keypoints(body1_joints, transformed_body1_joints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original body 1 joints
    ax.scatter(body1_joints[[indices], 0], body1_joints[[indices], 1], body1_joints[[indices], 2], c='b', label='Original Body 1')

    # Plot transformed body 1 joints
    ax.scatter(transformed_body1_joints[[indices], 0], transformed_body1_joints[[indices], 1], transformed_body1_joints[[indices], 2], c='g', label='Transformed Body 1')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


PATH = "data/demo_vid.pkl"
results = joblib.load(PATH)

body1_joints = results['outputs//_DEMO/vid/img/000180.jpg']["3d_joints"][0]  # Replace with actual keypoints

num_frames = 50

# Extract key joints and compute orientations
hip1, shoulder1, head1 = extract_key_joints(body1_joints)
# TODO automate this
hip2, shoulder2, head2 = np.asarray([0.1, 0, -0.375]), np.asarray([-0.2, 0, 0.1]), np.asarray([0, 0, 0.375])

orientation1 = compute_orientation(hip1, shoulder1, head1)
orientation2 = compute_orientation(hip2, shoulder2, head2)

# Compute the relative transformation matrix
relative_transformation = np.dot(orientation2, np.linalg.inv(orientation1))

mocap_keypoints = OrderedDict({
    "neck": 40,  # good
    "head": 18,  # good
    "torso": 27,
    "shoulder_right": 2,
    "shoulder_left": 34,
    "elbow_right": 3,
    "elbow_left": 6,
    "wrist_right": 31,  # good
    "wrist_left": 36, # good
    "knee_right":  26,  # good
    "knee_left":  29,  # good
    "foot_right":  21, # good
    "foot_left":  25,   # good
})

mocap_to_robot = OrderedDict({
    "neck": "neck",
    "head": "head",
    "shoulder_right": "right_shoulder",
    "shoulder_left": "left_shoulder",
    "elbow_right": "right_elbow",
    "elbow_left": "left_elbow",
    "knee_right": "right_knee",
    "knee_left": "left_knee",
})

indices = list(mocap_keypoints.values())
model, data, keypoints = load_model_get_keypoints()

def compute_average_translation(mocap_to_robot, model, data):
    diff = []
    for mocap_joint, sim_joint in mocap_to_robot.items():
        if sim_joint == "head":
            frame_id = model.getFrameId(sim_joint)
            simulator = data.oMf[frame_id].translation
            mocap = body1_joints[mocap_keypoints[mocap_joint]]
            sim_frame_mocap = apply_transformation(mocap, relative_transformation)
            diff.append(simulator - sim_frame_mocap)

    avg_translation = np.mean(diff, axis=0)
    return avg_translation

translation = compute_average_translation(mocap_to_robot, model, data)

three_d_joints = []
for key, item in results.items():
    three_d_joints.append(item["3d_joints"][0][indices])


def visualize_3d_joints(three_d_joints, keypoints=None, translation=None, add_joint_numbers=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_graph(num):
        ax.clear()
        ax.set_title(f'Frame {num + 1}')
        joints = three_d_joints[num]

        # # the transformation should be applied through the frame lvl
        joints = apply_transformation(joints, relative_transformation) + translation

        xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
        ax.scatter(xs, ys, zs, c='orange')

        if keypoints is not None:
            xs, ys, zs = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
            ax.scatter(xs, ys, zs, c="skyblue")

        if add_joint_numbers:
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                ax.text(x, y, z, str(i), color='red', fontsize=8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.view_init(elev=1, azim=70)

    # Create an animation
    ani = FuncAnimation(fig, update_graph, frames=NUM_FRAMES, interval=33)

    # Save the animation as a gif or display it
    ani.save('3d_joints_animation.gif', writer='imagemagick')
    plt.show()


# visualize_keyposes(three_d_joints + translation, keypoints)
visualize_3d_joints(three_d_joints, keypoints, translation)
