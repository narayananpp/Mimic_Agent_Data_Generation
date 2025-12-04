import mujoco
import numpy as np
import glfw
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import pickle
from pathlib import Path

# -------------------------------
# Save data as MimicKit motion
# -------------------------------
from enum import Enum
import pickle
import numpy as np

# This copies MimicKit's enum
class LoopMode(Enum):
    CLAMP = 0
    WRAP = 1

def quat_pos(q):
    """
    Ensure quaternion has positive scalar part (q[...,3]).
    q: [..., 4]
    returns q: [...,4]
    """
    q = np.asarray(q)
    sign = np.where(q[..., 3:] < 0, -1.0, 1.0)
    return q * sign


def quat_to_axis_angle(q, eps=1e-5):
    """
    Convert quaternion -> (axis, angle)
    q: [..., 4]  quaternion in [x,y,z,w] or [w,x,y,z]?
    
    YOUR ORIGINAL TORCH VERSION USES:
        q[..., qx:qw] = x,y,z
        q[..., qw]    = w
    So expected input layout = [x, y, z, w]   ← MUJOCO FORMAT
    """
    q = quat_pos(q)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    # vector part norm
    v = np.stack([x, y, z], axis=-1)
    length = np.linalg.norm(v, axis=-1)

    # angle
    angle = 2.0 * np.arctan2(length, w)

    # axis (safe divide)
    axis = np.zeros_like(v)
    mask = length > eps
    axis[mask] = v[mask] / length[mask][..., None]

    # default axis if angle is tiny
    default_axis = np.zeros_like(v)
    default_axis[..., 2] = 1.0
    axis[~mask] = default_axis[~mask]

    # zero angle for tiny rotations
    angle = np.where(mask, angle, np.zeros_like(angle))

    return axis, angle


def axis_angle_to_exp_map(axis, angle):
    """
    axis: [...,3]
    angle: [...]
    """
    angle = np.asarray(angle)[..., None]
    return axis * angle


def quat_to_exp_map(q):
    """
    Final mapping: quaternion -> exponential map
    q: [..., 4]  in MUJOCO format [x,y,z,w]
    """
    axis, angle = quat_to_axis_angle(q)
    return axis_angle_to_exp_map(axis, angle)


class MultiLinkGradientDescentIK:
    """Multi-link Gradient Descent IK solver using stacked Jacobians"""

    def __init__(self, model, data, body_names, step_size=0.01, tol=1e-3):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol

        # Get body IDs
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]

        # Get site IDs (foot tips)
        self.site_ids = []
        for name in body_names:
            # Guess foot site name
            site_name = name.replace("_calf", "_foot")
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                print(f"⚠️  Site {site_name} not found; using body {name} position")
                sid = None
            self.site_ids.append(sid)

        # Collect controllable DOFs (joint indices)
        self.dof_ids = self._get_leg_dofs()

        print(f"IK initialized: {len(self.dof_ids)} DOFs, {len(self.site_ids)} sites")

    def _get_leg_dofs(self):
        dof_ids = []
        for leg in ["FL", "FR", "RL", "RR"]:
            for joint in ["hip", "thigh", "calf"]:
                jn = f"{leg}_{joint}_joint"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    dof_ids.append(self.model.jnt_dofadr[jid])
        return np.array(dof_ids, dtype=int)

    def get_foot_positions(self):
        pos = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            if sid is not None:
                pos.append(self.data.site_xpos[sid].copy())
            else:
                pos.append(self.data.xpos[bid].copy())
        return np.array(pos)

    def damped_least_squares(self, J, dx, damping=1e-3):
        H = J.T @ J + (damping**2) * np.eye(J.shape[1])
        dq = np.linalg.solve(H, J.T @ dx)
        return dq

    def calculate(self, target_feet_pos, max_iter=500, damping=1e-3, debug=False):
        mujoco.mj_fwdPosition(self.model, self.data)
        dq = np.zeros(self.model.nv)

        for it in range(max_iter):
            current = self.get_foot_positions()
            err = (target_feet_pos - current).flatten()
            avg_err = np.linalg.norm(err) / len(self.site_ids)
            if avg_err < self.tol:
                break

            # Build stacked Jacobian
            J_list = []
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            for sid in self.site_ids:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
                J_list.append(jacp[:, self.dof_ids].copy())
            J = np.vstack(J_list)

            dq_local = self.step_size * self.damped_least_squares(J, err, damping)
            dq[self.dof_ids] = dq_local
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1)
            mujoco.mj_fwdPosition(self.model, self.data)

        if debug:
            print(f"[IK] Finished in {it} iters, avg_err={avg_err*1000:.2f} mm")

        return avg_err

class SkatingGaitController:
    def __init__(self, base_init_feet_pos, freq=1.0, 
                 push_ratio=0.3, recovery_ratio=0.3, glide_ratio=0.4,
                 step_length=0.12, step_height=0.05):
        
        self.freq = freq
        self.push_ratio = push_ratio
        self.recovery_ratio = recovery_ratio
        self.glide_ratio = glide_ratio
        
        self.step_length = step_length
        self.step_height = step_height

        self.base_init = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()
        
        self.cycle_period = 1.0 / self.freq

    def set_base_init_feet_pos(self, vx=1.0, dt=0.002):
        """Shift reference foot positions forward as the body moves."""
        for leg in self.base_feet_pos.keys():
            self.base_feet_pos[leg] += np.array([vx, 0, 0]) * dt

    def _compute_leg_phase(self, leg_name, t):
        """
        Compute which phase the leg is in for alternating back-leg gait.
        """
        if leg_name in ["FL_calf", "FR_calf"]:
            return 0.0, False
        
        period = 2.0 / self.freq
        time_in_period = t % period
        cycle_time = 1.0 / self.freq
        
        if leg_name == "RL_calf": # or leg_name == "FL_calf":
            if time_in_period < cycle_time:
                phase = time_in_period / cycle_time
                should_execute = True
            else:
                phase = 1.0
                should_execute = False
                
        elif leg_name == "RR_calf": # or leg_name == "FR_calf":
            if time_in_period >= cycle_time:
                phase = (time_in_period - cycle_time) / cycle_time
                should_execute = True
            else:
                phase = 1.0
                should_execute = False
        else:
            phase = 0.0
            should_execute = False
        
        return phase, should_execute

    def foot_target(self, leg_name, t):
        """
        Calculate target foot position based on alternating skating gait.
        """
        phase, should_execute = self._compute_leg_phase(leg_name, t)
        
        foot = self.base_feet_pos[leg_name].copy()
        
        # DEBUG: Print status
        if leg_name in ["RL_calf", "RR_calf"]:
            phase_name = "GLIDE"
            if should_execute:
                if phase < self.push_ratio:
                    phase_name = "PUSH"
                elif phase < self.push_ratio + self.recovery_ratio:
                    phase_name = "RECOVERY"
            
            # print(f"t={t:.2f} {leg_name}: phase={phase:.3f}, execute={should_execute}, {phase_name}")
        
        if not should_execute:
            return foot
        
        # PUSH PHASE
        if phase < self.push_ratio:
            progress = phase / self.push_ratio
            foot[0] = self.base_feet_pos[leg_name][0] - self.step_length * progress
            foot[2] = self.base_feet_pos[leg_name][2]
            return foot

        # RECOVERY PHASE
        elif phase < self.push_ratio + self.recovery_ratio:
            phase_rel = phase - self.push_ratio
            progress = phase_rel / self.recovery_ratio
            angle = np.pi * progress
            
            foot[0] = (
                self.base_feet_pos[leg_name][0]
                - self.step_length * (1 - progress)
            )
            foot[2] = (
                self.base_feet_pos[leg_name][2]
                + self.step_height * np.sin(angle)
            )
            return foot

        # GLIDE PHASE
        else:
            return self.base_feet_pos[leg_name]


class WalkingGaitController:
    def __init__(self, base_init_feet_pos, freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05):
        self.freq = freq
        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        self.base_init_feet_pos = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()
        
        # Phase offsets (radians)
        self.phase_offsets = {
            "_calf_calf": 0.0,
            "FR_calf": np.pi,
            "RL_calf": np.pi,
            "RR_calf": 0.0,
        }
    
    def set_base_init_feet_pos(self, vx =1, dt=0.002):
        for feet in ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]:
            self.base_feet_pos[feet] += np.array([vx, 0, 0]) * dt


    def foot_target(self, leg_name, t, mode="moving"):
        """Compute desired foot position for leg_name at time t"""
        phi = 2 * np.pi * self.freq * t + self.phase_offsets[leg_name]
        phase = (phi % (2*np.pi)) / (2*np.pi)
        foot = self.base_feet_pos[leg_name].copy()
        
        if mode=="moving":
            if phase < self.duty:
                # --- Stance phase (foot on ground, moves backward relative to body)
                progress = phase / self.duty
                foot[0] -= self.step_length * (progress - 0.5)
                foot[2] = self.base_feet_pos[leg_name][2]  # stay on ground
            else:
                # --- Swing phase (foot in air following semicircle)
                progress = (phase - self.duty) / (1 - self.duty)
                angle = np.pi * progress
                foot[0] += self.step_length * (progress - 0.5)
                foot[2] = self.base_feet_pos[leg_name][2] + self.step_height * np.sin(angle)
        else:
            foot =  self.base_feet_pos[leg_name]

        return foot

# ----------------------------
# Utility: body frame transform
# ----------------------------
def world_to_body(model, data, point_world):
    """Convert a 3D point from world frame to body frame."""
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    R_wb = data.xmat[trunk_id].reshape(3, 3)  # rotation from body to world
    p_wb = data.xpos[trunk_id]
    return R_wb.T @ (point_world - p_wb)  # inverse rotation + translation removal

def compute_phase(t, freq):
    """Phase ∈ [0, 1] for gait cycle."""
    return (freq * t) % 1.0

# ============================================================
#                MAIN FUNCTION — CLEAN VERSION
# ============================================================
def main(mode="skating", robot="go2"):
    # Load model (pure MJCF, no physics necessary)
    model = mujoco.MjModel.from_xml_path(f"assets/unitree_{robot}/scene.xml")
    data = mujoco.MjData(model)

    # Initialize GLFW window
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW.")
    window = glfw.create_window(1280, 900, "Go2 Kinematic IK Demo", None, None)
    glfw.make_context_current(window)

    # Visualization setup
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.3]
    cam.distance = 2.5
    cam.azimuth, cam.elevation = 90, -20
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    
    # Reset to home pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    
    # foot_length = 0.6
    # data.qpos[0:3] = [0, 0, foot_length]
    # data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_fwdPosition(model, data)

    # Initialize IK
    body_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    ik = MultiLinkGradientDescentIK(model, data, body_names)

    initial_feet_pos = ik.get_foot_positions()
    # initial_feet_pos[:, 2] = 0.1
    print("Initial foot positions:")
    for i, n in enumerate(body_names):
        print(f"  {n}: {initial_feet_pos[i]}")

    
    frame = 0
    
    # Convert to dictionary for easy access
    base_feet = {n: initial_feet_pos[i].copy() for i, n in enumerate(body_names)}
    # base_feet["FL_calf"][2] = base_feet["FR_calf"][2] = base_feet["RL_calf"][2] = base_feet["RR_calf"][2] = 0.0
        
    # --------------------------------
    # Live Matplotlib setup
    # --------------------------------
    plt.ion()
    fig, (ax_z, ax_x) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = ["r", "g", "b", "m"]
    lines_z, lines_x = [], []

    # Buffers (fixed-size deques for real-time)
    buffer_len = 100
    total_time = 0.5
    time_buffer = deque(np.linspace(0, total_time, buffer_len), maxlen=buffer_len)
    pos_z_buffers = [deque(np.zeros(buffer_len), maxlen=buffer_len) for _ in body_names]
    pos_x_buffers = [deque(np.zeros(buffer_len), maxlen=buffer_len) for _ in body_names]

    # Z plot setup
    for i, name in enumerate(body_names):
        line_z, = ax_z.plot(time_buffer, pos_z_buffers[i], color=colors[i], label=f"{name} Z")
        lines_z.append(line_z)
    ax_z.set_xlim(0, total_time)
    ax_z.set_ylim(-0.4, 0.4)
    ax_z.set_ylabel("Foot Z height (m)")
    ax_z.legend(loc="upper right")

    # X plot setup
    for i, name in enumerate(body_names):
        line_x, = ax_x.plot(time_buffer, pos_x_buffers[i], color=colors[i], linestyle="--", label=f"{name} X")
        lines_x.append(line_x)
    ax_x.set_xlim(0, total_time)
    ax_x.set_ylim(-0.5, 0.5)
    ax_x.set_xlabel("Time (s)")
    ax_x.set_ylabel("Foot X position (m)")
    ax_x.legend(loc="upper right")

    plt.tight_layout()
    plt.show(block=False)
    
    # --------------------------------
    # Define gait settings
    # --------------------------------
    base_velocity = 0.6
    step_length = 0.3
    gait_freq = base_velocity / step_length

    if mode=="walking":
        print("Walking Gait Controller Initialized")
        gait = WalkingGaitController(
                base_init_feet_pos=base_feet,
                freq=gait_freq,
                duty_ratio=0.75,
                step_length=step_length,
                step_height=0.1
            )
    elif  mode=="skating":
        print("Skating Gait Controller Initialized")
        gait = SkatingGaitController(
                    base_init_feet_pos=base_feet,
                    freq=gait_freq,
                    step_length=step_length,
                    step_height=0.15
                )
    # --------------------------------
    # One gait cycle duration
    # --------------------------------
    T_cycle = 1.0 / gait_freq

    # --------------------------------
    # Simulation timing
    # --------------------------------
    sim_freq = 200.0          # Hz
    dt = 1.0 / sim_freq
    FPS = int(sim_freq)       # stored in MimicKit file

    # --------------------------------
    # Storage
    # --------------------------------
    frames_qpos = []          # store raw MuJoCo qpos
    frames_root_pos = []
    frames_root_exp = []
    frames_joint = []
    frames_foot_body = []
    frames_phase = []

    t = 0.0
    start_time = 0.5
    end_time = 2*T_cycle + start_time
    # -------------------------------
    # Main simulation loop
    # -------------------------------
    while not glfw.window_should_close(window):
        target_feet_pos = initial_feet_pos.copy()
        qpos = data.qpos.copy()

        gait.set_base_init_feet_pos(vx=base_velocity, dt=dt)
        
        for i, n in enumerate(body_names):
            if mode=="skating":               
                target_feet_pos[i] = gait.foot_target(n, t)
                
            elif mode=="walking":
                target_feet_pos[i] = gait.foot_target(n, t, mode="moving")
            else:
                target_feet_pos[i] = gait.foot_target(n, t, mode="static")


        err = ik.calculate(target_feet_pos, debug=(frame % 60 == 0))

        # --- Collect data for plot ---
        feet = ik.get_foot_positions()  # shape (4, 3)

        for i in range(4):
            pos_z_buffers[i].append(feet[i, 2])
            pos_x_buffers[i].append(feet[i, 0])

        # --- Update plot every few frames ---
        if frame % 2 == 0:
            for i in range(4):
                lines_z[i].set_ydata(pos_z_buffers[i])
                lines_x[i].set_ydata(pos_x_buffers[i])
            plt.pause(0.001)
        
        
        # --------------------------------
        # Collect data
        # --------------------------------
        qpos = data.qpos.copy()
        frames_qpos.append(qpos)

        # Root pose
        root_pos = qpos[0:3] + np.array([base_velocity, 0, 0]) * dt
        # Convert MuJoCo (w,x,y,z) → MimicKit order (x,y,z,w)
        mj_quat = qpos[3:7]        # [w,x,y,z]
        root_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])
        root_exp = quat_to_exp_map(root_quat)
        
        if t >= start_time:
            frames_root_pos.append(root_pos)
            frames_root_exp.append(root_exp)
            # Joint angles (everything after the floating base)
            frames_joint.append(qpos[7:].copy())

        # --- Render simulation ---
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 1280, 900), scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

        if t >= end_time:  
            print(f"Recorded One Gait Cycle")
            break

        # --- Advance time with fixed dt ---
        frame += 1
        t += dt
        data.qpos[0:3] += np.array([base_velocity, 0, 0]) * dt
        glfw.wait_events_timeout(dt)  # maintain ~200Hz wall-time update rate

    glfw.terminate()
    # ============================================================
    #        BUILD MimicKit Frames: [pos(3), expmap(3), joints]
    # ============================================================
    frames = []
    for pos, exp, joint in zip(frames_root_pos, frames_root_exp, frames_joint):
        frame = np.concatenate([pos, exp, joint])
        frames.append(frame)

    frames = np.vstack(frames)   # shape [T, D]

    # ============================================================
    #        SAVE MimicKit motion file
    # ============================================================
    motion = dict(
        fps=FPS,
        loop_mode=1,  # WRAP
        frames=frames
        # phase=np.array(frames_phase),         # optional but useful
        # foot_body=np.array(frames_foot_body)  # optional debugging
    )

    save_path = Path(f"data/{robot}_{mode}_gait_mimickit.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(motion, f)

    print(f"✅ Saved 1-cycle MimicKit motion: {save_path}")
    print(f"   frames: {frames.shape}, fps={FPS}")
    print("   One gait cycle duration:", T_cycle)
    
if __name__ == "__main__":
    main(mode="skating", robot="go2w")
