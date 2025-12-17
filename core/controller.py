import numpy as np

from gaits import get_gait_controller
from utils.kinematics import MultiLinkGradientDescentIK
from utils.math_utils import (
    quat_to_euler,
    integrate_yaw,
    delta_vector,
)
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GaitControllerRunner:
    """
    Orchestrates:
    - Gait generation
    - IK solving
    - Base motion integration
    """

    def __init__(self, sim, args):
        self.sim = sim
        self.args = args
        self.model = sim.model
        self.data = sim.data

        # -----------------------------
        # Feet / IK
        # -----------------------------
        self.body_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

        self.ik = MultiLinkGradientDescentIK(
            self.model,
            self.data,
            self.body_names,
        )

        self.base_feet = {
            n: p.copy()
            for n, p in zip(self.body_names, self.ik.get_foot_positions())
        }

        # -----------------------------
        # Gait (via registry)
        # -----------------------------
        Gait = get_gait_controller(args.mode)
        self.gait = Gait(
            base_init_feet_pos=self.base_feet,
            freq=args.gait_freq,
            step_length=args.step_length,
            step_height=args.step_height,
            style=getattr(args, "skating_style", None)
        )

        # -----------------------------
        # Timing
        # -----------------------------
        self.dt = 1.0 / args.sim_freq
        self.t = 0.0
        self.frame = 0

    # =====================================================
    #                     STEP
    # =====================================================
    def step(self):
        """
        Advances simulation by one control step.

        Returns:
            dict with root + joint state for recording
        """

        # ---------------------------------
        # Read current base orientation
        # ---------------------------------
        qpos = self.data.qpos
        _, _, yaw = quat_to_euler(qpos[3:7])

        # ---------------------------------
        # Update gait base motion
        # ---------------------------------
        self.gait.set_base_init_feet_pos(
            vx=self.args.base_velocity,
            yaw=yaw,
            dt=self.dt,
            yaw_rate=self.args.yaw_rate,
        )

        # ---------------------------------
        # Foot targets
        # ---------------------------------
        targets = np.zeros((4, 3))
        for i, leg in enumerate(self.body_names):
            targets[i] = self.gait.foot_target(leg, self.t)

        # ---------------------------------
        # IK solve
        # ---------------------------------
        self.ik.calculate(targets, debug=(self.frame % 60 == 0))

        # ---------------------------------
        # Integrate floating base
        # ---------------------------------
        root_pos = qpos[0:3] + delta_vector(
            self.args.base_velocity,
            yaw,
            self.dt,
            self.args.yaw_rate,
        )

        root_quat = integrate_yaw(
            qpos[3:7],
            self.args.yaw_rate,
            self.dt,
        )

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat

        # ---------------------------------
        # Advance time
        # ---------------------------------
        self.t += self.dt
        self.frame += 1
        # ---------------------------------
        # Return recorder-friendly state
        # ---------------------------------
        return {
            "time": self.t,
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),  # [w,x,y,z]
            "joints": self.data.qpos[7:].copy(),
        }
