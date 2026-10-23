"""Student functions for HW6: trajectories and integration in SE(3)."""

import numpy as np

from hw2.kin_func_skeleton import inverse_SE3, renormalize_SE3, se3_to_twist, twist_to_SE3


L1 = 0.57
L2 = 0.25
L3 = 0.20
L4 = 0.10
K = 0.5


def trajectory(t):
    """Return the chair-swing pose defined in the handout.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing g_ab(t) in the ride
        frame.
    """
    phi = K * np.sin(t)

    pt1 = np.array([-L2 * np.sin(t), L2 * np.cos(t), L1 + L4 - L4 * np.cos(t)])
    pt2 = np.array([-L3 * np.sin(t) * np.sin(phi), L3 * np.cos(t) * np.sin(phi), -L3 * np.cos(phi)])
    p = pt1 + pt2

    Bx = np.array([-np.cos(t), -np.sin(t), 0.0])
    Bz = np.array([np.sin(phi) * np.sin(t), -np.sin(phi) * np.cos(t), np.cos(phi)])
    By = np.cross(Bz, Bx)

    g = np.eye(4)
    g[:3, :3] = np.column_stack((Bx, By, Bz))
    g[:3, 3] = p
    return g


def body_velocity(t):
    """Calculate the instantaneous body velocity of the chair swing.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing the body velocity
        matrix in se(3). Analytically differentiate the full trajectory
        matrix, including its rotation, and compute the body velocity.

    inverse_SE3 from kin_func_skeleton inverts the pose.
    """
    phi = K * np.sin(t)
    phi_dot = K * np.cos(t)

    pdot1 = np.array([-L2 * np.cos(t), -L2 * np.sin(t), L4 * np.sin(t)])
    pdot2 = np.array([-L3 * np.cos(t) * np.sin(phi) - L3 * np.sin(t) * np.cos(phi) * phi_dot, -L3 * np.sin(t) * np.sin(phi) + L3 * np.cos(t) * np.cos(phi) * phi_dot, L3 * np.sin(phi) * phi_dot])
    pdot = pdot1 + pdot2

    R_dot1a = np.sin(t)
    R_dot1b = np.cos(t) * np.cos(phi) - phi_dot * np.sin(t) * np.sin(phi)
    R_dot1c = np.cos(t) * np.sin(phi) + phi_dot * np.sin(t) * np.cos(phi)
    R_dot1 = [R_dot1a, R_dot1b, R_dot1c]

    R_dot2a = -np.cos(t)
    R_dot2b = np.sin(t) * np.cos(phi) + phi_dot * np.cos(t) * np.sin(phi)
    R_dot2c = np.sin(t) * np.sin(phi) - phi_dot * np.cos(t) * np.cos(phi)
    R_dot2 = [R_dot2a, R_dot2b, R_dot2c]

    R_dot3a = 0.0
    R_dot3b = -phi_dot * np.cos(phi)
    R_dot3c = -phi_dot * np.sin(phi)
    R_dot3 = [R_dot3a, R_dot3b, R_dot3c]
    R_dot = np.array([R_dot1, R_dot2, R_dot3])

    g_dot = np.zeros((4, 4))
    g_dot[:3, :3] = R_dot
    g_dot[:3, 3] = pdot

    # V^b =
    return inverse_SE3(trajectory(t)) @ g_dot


def euler_step(pose, velocity_hat, dt):
    """Advance an SE(3) pose with body velocity held constant for one step.

    Args:
        pose: NumPy array with shape (4, 4), the current pose.
        velocity_hat: NumPy array with shape (4, 4), a body velocity in se(3).
        dt: Nonnegative scalar timestep in seconds.

    Returns:
        NumPy array with shape (4, 4), the next SE(3) pose. Apply the
        exponential rigid-motion increment on the right of the current
        pose, as in Problem 4.

    se3_to_twist converts the velocity matrix to a vector. twist_to_SE3
    computes the rigid-motion increment over dt. Apply renormalize_SE3
    to the updated pose to correct floating-point drift in its rotation.
    """
    xi = se3_to_twist(velocity_hat)
    return renormalize_SE3(pose @ twist_to_SE3(xi, dt))
