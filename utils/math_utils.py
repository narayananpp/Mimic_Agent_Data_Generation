import numpy as np

def quat_to_euler(q):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw)
    """
    w, x, y, z = q
    # Roll
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)
    # Pitch
    t2 = +2.0 * (w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    # Yaw
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw

def euler_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion [w, x, y, z]
    """
    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def delta_vector(vx, theta, dt, yaw_rate=0.0):
    """Compute delta position for body movement."""
    if abs(yaw_rate) < 1e-6:
        del_x = vx * np.cos(theta) * dt
        del_y = vx * np.sin(theta) * dt
    else:
        del_x = (vx / yaw_rate) * (np.sin(theta + yaw_rate * dt) - np.sin(theta))
        del_y = -(vx / yaw_rate) * (np.cos(theta + yaw_rate * dt) - np.cos(theta))
    return np.array([del_x, del_y, 0.0])


def integrate_yaw(quat, yaw_rate, dt):
    """
    Integrate yaw into a quaternion assuming rotation about world Z axis.

    Args:
        quat (np.ndarray): current quaternion [w, x, y, z] (MuJoCo format)
        yaw_rate (float): yaw rate (rad/s)
        dt (float): timestep (s)

    Returns:
        np.ndarray: updated quaternion [w, x, y, z]
    """
    yaw_delta = yaw_rate * dt

    if abs(yaw_delta) < 1e-8:
        return quat.copy()

    # Incremental yaw quaternion (about Z)
    half = 0.5 * yaw_delta
    q_yaw = np.array([
        np.cos(half),  # w
        0.0,           # x
        0.0,           # y
        np.sin(half)   # z
    ])

    # Quaternion multiplication: q_new = q_yaw ⊗ quat
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = quat

    q_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    # Normalize to avoid drift
    q_new /= np.linalg.norm(q_new)

    return q_new


def quat_to_exp_map(quat):
    """
    Convert a quaternion to exponential map (axis-angle) representation.

    Args:
        quat: array-like, shape (4,) quaternion [x, y, z, w]

    Returns:
        exp_map: np.array shape (3,) exponential map vector
    """
    x, y, z, w = quat
    # Normalize quaternion
    norm = np.linalg.norm([x, y, z, w])
    if norm > 0:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    else:
        return np.zeros(3)

    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))

    # Avoid division by zero
    s = np.sqrt(1 - w * w)
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])  # Arbitrary axis
    else:
        axis = np.array([x, y, z]) / s

    exp_map = axis * angle
    return exp_map


