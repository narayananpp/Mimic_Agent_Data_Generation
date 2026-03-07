import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_euler(q):
    """Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(+2.0 * (w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def quat_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    Used by controller.py to transform foot positions between frames.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def body_to_world_position(pos_body, root_pos, root_quat):
    """
    Transform a position from body frame to world frame.

    Args:
        pos_body  : np.ndarray (3,) — position in body frame
        root_pos  : np.ndarray (3,) — root position in world frame
        root_quat : np.ndarray (4,) — root orientation [w, x, y, z]

    Returns:
        np.ndarray (3,) — position in world frame
    """
    R_mat = quat_to_rotation_matrix(root_quat)
    return root_pos + R_mat @ pos_body


def world_to_body_position(pos_world, root_pos, root_quat):
    """
    Transform a position from world frame to body frame.

    Args:
        pos_world : np.ndarray (3,) — position in world frame
        root_pos  : np.ndarray (3,) — root position in world frame
        root_quat : np.ndarray (4,) — root orientation [w, x, y, z]

    Returns:
        np.ndarray (3,) — position in body frame
    """
    R_mat = quat_to_rotation_matrix(root_quat)
    return R_mat.T @ (pos_world - root_pos)


def integrate_pose_world_frame(root_pos, root_quat, vel_world, omega_world, dt):
    """
    Integrate root pose given velocities expressed in WORLD frame.

    Args:
        root_pos    : np.ndarray (3,)  — current position [x, y, z]
        root_quat   : np.ndarray (4,)  — current orientation [w, x, y, z]
        vel_world   : np.ndarray (3,)  — linear velocity in world frame (m/s)
        omega_world : np.ndarray (3,)  — angular velocity in world frame (rad/s)
        dt          : float            — timestep (s)

    Returns:
        new_pos  : np.ndarray (3,)
        new_quat : np.ndarray (4,)  [w, x, y, z]
    """
    # ── Position integration ──────────────────────────────────
    new_pos = root_pos + vel_world * dt

    # ── Orientation integration ───────────────────────────────
    angle = np.linalg.norm(omega_world * dt)
    if angle < 1e-8:
        new_quat = root_quat.copy()
    else:
        axis = omega_world / np.linalg.norm(omega_world)
        delta_r = R.from_rotvec(axis * angle)
        dq = delta_r.as_quat()          # scipy: [x, y, z, w]
        dq_wxyz = np.array([dq[3], dq[0], dq[1], dq[2]])   # → [w, x, y, z]

        # q_new = dq ⊗ q  (world-frame rotation applied on the left)
        w1, x1, y1, z1 = dq_wxyz
        w2, x2, y2, z2 = root_quat
        new_quat = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        new_quat /= np.linalg.norm(new_quat)

    return new_pos, new_quat


def delta_pose(linear_vel, angular_vel, quat, dt):
    """
    Integrate 6-DOF motion from BODY-frame velocities.

    Args:
        linear_vel  : np.ndarray (3,) — velocity in body frame (m/s)
        angular_vel : np.ndarray (3,) — angular velocity in body frame (rad/s)
        quat        : np.ndarray (4,) — current orientation [w, x, y, z]
        dt          : float

    Returns:
        delta_pos : np.ndarray (3,)
        new_quat  : np.ndarray (4,) [w, x, y, z]
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy [x,y,z,w]
    delta_pos = r.apply(linear_vel) * dt

    angle = np.linalg.norm(angular_vel * dt)
    if angle < 1e-8:
        delta_quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = angular_vel / np.linalg.norm(angular_vel)
        dq = R.from_rotvec(axis * angle).as_quat()         # [x,y,z,w]
        delta_quat = np.array([dq[3], dq[0], dq[1], dq[2]])

    w1, x1, y1, z1 = delta_quat
    w2, x2, y2, z2 = quat
    q_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    q_new /= np.linalg.norm(q_new)
    return delta_pos, q_new


def integrate_yaw(quat, yaw_rate, dt):
    """Integrate yaw about world Z axis into a quaternion [w, x, y, z]."""
    yaw_delta = yaw_rate * dt
    if abs(yaw_delta) < 1e-8:
        return quat.copy()
    half = 0.5 * yaw_delta
    q_yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = quat
    q_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    q_new /= np.linalg.norm(q_new)
    return q_new


def quat_to_exp_map(quat):
    """Convert quaternion [w, x, y, z] to exponential map (axis-angle)."""
    w, x, y, z = quat
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-8:
        return np.zeros(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(0.0, 1.0 - w*w))
    axis = np.array([x, y, z]) / s if s > 1e-8 else np.array([1.0, 0.0, 0.0])
    return axis * angle