# gaits/static_gait.py
import numpy as np
from gaits.base import BaseGaitController
from utils.math_utils import delta_vector

class StaticGaitController(BaseGaitController):
    def __init__(self, base_init_feet_pos, freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05, style="handstand"):
        super().__init__(base_init_feet_pos, freq)
        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        self.base_feet_pos = base_init_feet_pos.copy()

        # Phase offsets (radians)
        self.phase_offsets = {
            "FL_calf": 0.0,
            "FR_calf": np.pi,
            "RL_calf": np.pi,
            "RR_calf": 0.0,
        }

    def set_base_init_feet_pos(self, vx=1.0, yaw=0, dt=0.002, yaw_rate=0.0):
        """Shift reference foot positions forward as the body moves."""
        for leg in self.base_feet_pos.keys():
            self.base_feet_pos[leg] += delta_vector(vx=vx, theta=yaw, dt=dt, yaw_rate=yaw_rate)

    def foot_target(self, leg_name, t, **kwargs):
        phi = 2 * np.pi * self.freq * t + self.phase_offsets[leg_name]
        phase = (phi % (2*np.pi)) / (2*np.pi)
        foot = self.base_feet_pos[leg_name].copy()
        return foot
