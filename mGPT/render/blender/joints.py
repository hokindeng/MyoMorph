import math

import bpy
import numpy as np

from mGPT.utils.joints import (myoskeleton_joints, myoskeleton_kinematic_tree)

# from .materials import colored_material_diffuse_BSDF as colored_material
from .materials import colored_material_relection_BSDF as colored_material

sat_factor = 1.1

JOINTS_MATS = [
    # colored_material(0.2500, 0.0357, 0.0349, saturation_factor = sat_factor),
    # # colored_material(0.4500, 0.0357, 0.0349),
    # colored_material(0.6500, 0.175, 0.0043, saturation_factor = sat_factor),
    # colored_material(0.0349, 0.3500, 0.0349, saturation_factor = sat_factor),
    # colored_material(0.018, 0.059, 0.600, saturation_factor = sat_factor),
    # colored_material(0.032, 0.325, 0.421, saturation_factor = sat_factor),
    # colored_material(0.3, 0.3, 0.3, saturation_factor = sat_factor),
    colored_material(0.3500, 0.0357, 0.0349, saturation_factor=sat_factor),
    # colored_material(0.4500, 0.0357, 0.0349),
    colored_material(0.6500, 0.175, 0.0043, saturation_factor=sat_factor),
    colored_material(0.0349, 0.3500, 0.0349, saturation_factor=sat_factor),
    colored_material(0.018, 0.059, 0.600, saturation_factor=sat_factor),
    colored_material(0.032, 0.325, 0.421, saturation_factor=sat_factor),
    colored_material(0.3, 0.3, 0.3, saturation_factor=sat_factor),
]


class Joints:

    def __init__(self,
                 data,
                 *,
                 mode,
                 canonicalize,
                 always_on_floor,
                 **kwargs):
        data = prepare_joints(
            data,
            canonicalize=canonicalize,
            always_on_floor=always_on_floor
        )

        self.data = data
        self.mode = mode

        self.N = len(data)

        self.N = len(data)
        self.trajectory = data[:, 0, [0, 1]]

        self.kinematic_tree = myoskeleton_kinematic_tree
        self.joints = myoskeleton_joints

        self.mat = JOINTS_MATS

    def get_sequence_mat(self, frac):
        return self.mat

    def get_root(self, index):
        return self.data[index][0]

    def get_mean_root(self):
        return self.data[:, 0].mean(0)

    def load_in_blender(self, index, mats):
        skeleton = self.data[index]
        head_mat = mats[0]
        body_mat = mats[-1]
        for lst, mat in zip(self.kinematic_tree, mats):
            for j1, j2 in zip(lst[:-1], lst[1:]):
                # spine and head
                if self.joints[j2] in [
                        "skull",
                ]:
                    sphere_between(skeleton[j1], skeleton[j2], head_mat)
                elif self.joints[j2] in [
                        "ulna_l",
                        "ulna_r",
                        "lunate_l",
                        "lunate_r",
                ]:
                    cylinder_sphere_between(skeleton[j1], skeleton[j2], 0.040,
                                            mat)
                elif self.joints[j2] in [
                        "talus_l",
                        "talus_r",
                        "tibia_r",
                        "tibia_l",
                ]:
                    cylinder_sphere_between(skeleton[j1], skeleton[j2], 0.040,
                                            mat)
                elif self.joints[j2] in [
                        "clavicle_l",
                        "clavicle_r",
                        "foot_l",
                        "foot_r",
                ]:
                    cylinder_between(skeleton[j1], skeleton[j2], 0.040, mat)
                elif self.joints[j2] in ["tibia_r", "tibia_l"]:
                    print(self.joints[j1], self.joints[j2])
        # body
        sphere(0.14, skeleton[self.joints.index("cervical_spine_1")], body_mat)
        sphere_between(
            skeleton[self.joints.index("cervical_spine_1")],
            skeleton[self.joints.index("myoskeleton_root")],
            body_mat,
            factor=0.28,
        )
        sphere(0.11, skeleton[self.joints.index("myoskeleton_root")], body_mat)
        # sphere_between(
        #     skeleton[self.joints.index("BLN")],
        #     skeleton[self.joints.index("BT")],
        #     mats[0],
        # )
        # hip
        # sphere_between(
        #     skeleton[self.joints.index("LH")],
        #     skeleton[self.joints.index("RH")],
        #     mats[0],
        #     factor=0.6,
        # )
        #
        # sphere(skeleton[self.joints.index("BLN")], 0.05, mats[0])
        # sphere_between(skeleton[13], skeleton[14], mat)
        # node
        # print(self.joints.index("BUN"))
        # print(len(lst))
        # sphere(lst[self.joints.index("BUN")], 0.2, mat)  # head

        return ["Cylinder", "Sphere"]

    def __len__(self):
        return self.N


def softmax(x, softness=1.0, dim=None):
    maxi, mini = x.max(dim), x.min(dim)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, softness=1.0, dim=0):
    return -softmax(-x, softness=softness, dim=dim)


def get_forward_direction(poses, jointstype="myoskeleton"):
    if jointstype == "myoskeleton":
        joints = myoskeleton_joints
    else:
        raise TypeError("Only supports myoskeleton jointstype")
    # Shoulders
    LS, RS = joints.index("clavicle_l"), joints.index("clavicle_r")
    # Hips
    LH, RH = joints.index("hip_l"), joints.index("hip_r")

    across = (poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] -
              poses[..., LS, :])
    forward = np.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = forward / np.linalg.norm(forward, axis=-1)
    return forward


def cylinder_between(t1, t2, r, mat):
    x1, y1, z1 = t1
    x2, y2, z2 = t2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(radius=r,
                                        depth=dist,
                                        location=(dx / 2 + x1, dy / 2 + y1,
                                                  dz / 2 + z1))

    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat

    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x1, y1, z1))
    bpy.context.object.active_material = mat
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x2, y2, z2))
    bpy.context.object.active_material = mat


def cylinder_sphere_between(t1, t2, r, mat):
    x1, y1, z1 = t1
    x2, y2, z2 = t2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)
    dist = dist - 0.2 * r
    # sphere node
    sphere(r * 0.9, t1, mat)
    sphere(r * 0.9, t2, mat)
    # leveled cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=r,
        depth=dist,
        location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1),
        enter_editmode=True,
    )
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_face_by_sides(number=32, extend=False)
    bpy.ops.mesh.bevel(offset=r, segments=8)
    bpy.ops.object.editmode_toggle(False)
    # bpy.ops.object.shade_smooth()
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    bpy.context.object.active_material = mat


def sphere(r, t, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=50,
                                         ring_count=50,
                                         radius=r,
                                         location=t)
    # bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=t)
    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat


def sphere_between(t1, t2, mat, factor=1):
    x1, y1, z1 = t1
    x2, y2, z2 = t2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2) * factor

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=50,
        ring_count=50,
        # bpy.ops.mesh.primitive_uv_sphere_add(
        radius=dist,
        location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1))

    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat


def matrix_of_angles(cos, sin, inv=False):
    sin = -sin if inv else sin
    return np.stack((np.stack(
        (cos, -sin), axis=-1), np.stack((sin, cos), axis=-1)),
                    axis=-2)


def get_floor(poses, jointstype="myoskeleton"):
    if jointstype == "myoskeleton":
        joints = myoskeleton_joints
    else:
        raise TypeError("Only supports myoskeleton jointstype")
    # Feet
    LF, RF = joints.index("foot_l"), joints.index("foot_r")
    ndim = len(poses.shape)

    foot_heights = poses[..., (LF, RF), 1].min(-1)
    floor_height = softmin(foot_heights, softness=0.5, dim=-1)
    return floor_height[tuple((ndim - 2) * [None])].T


def canonicalize_joints(joints):
    poses = joints.copy()

    translation = joints[..., 0, :].copy()

    # Let the root have the Y translation
    translation[..., 1] = 0
    # Trajectory => Translation without gravity axis (Y)
    trajectory = translation[..., [0, 2]]

    # Remove the floor
    poses[..., 1] -= get_floor(poses)

    # Remove the trajectory of the joints
    poses[..., [0, 2]] -= trajectory[..., None, :]

    # Let the first pose be in the center
    trajectory = trajectory - trajectory[..., 0, :]

    # Compute the forward direction of the first frame
    forward = get_forward_direction(poses[..., 0, :, :])

    # Construct the inverse rotation matrix
    sin, cos = forward[..., 0], forward[..., 1]
    rotations_inv = matrix_of_angles(cos, sin, inv=True)

    # Rotate the trajectory
    trajectory_rotated = np.einsum("...j,...jk->...k", trajectory,
                                   rotations_inv)

    # Rotate the poses
    poses_rotated = np.einsum("...lj,...jk->...lk", poses[..., [0, 2]],
                              rotations_inv)
    poses_rotated = np.stack(
        (poses_rotated[..., 0], poses[..., 1], poses_rotated[..., 1]), axis=-1)

    # Re-merge the pose and translation
    poses_rotated[..., (0, 2)] += trajectory_rotated[..., None, :]
    return poses_rotated


def prepare_joints(joints,
                   canonicalize=True,
                   always_on_floor=False):
    # All face the same direction for the first frame
    if canonicalize:
        data = canonicalize_joints(joints)
    else:
        data = joints

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # Center the first root to the first frame
    data -= data[[0], [0], :]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data


def NormalInDirection(normal, direction, limit=0.5):
    return direction.dot(normal) > limit


def GoingUp(normal, limit=0.5):
    return NormalInDirection(normal, (0, 0, 1), limit)


def GoingDown(normal, limit=0.5):
    return NormalInDirection(normal, (0, 0, -1), limit)


def GoingSide(normal, limit=0.5):
    return GoingUp(normal, limit) == False and GoingDown(normal,
                                                         limit) == False
