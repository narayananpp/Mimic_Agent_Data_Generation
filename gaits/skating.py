import numpy as np
from gaits.base import BaseGaitController
from utils.math_utils import delta_vector

class SkatingGaitController(BaseGaitController):
    def __init__(self, base_init_feet_pos, freq=1.0,
                 push_ratio=0.2, recovery_ratio=0.2, glide_ratio=0.6,
                 step_length=0.12, step_height=0.05,
                 style="handstand"):
        super().__init__(base_init_feet_pos, freq)
        self.push_ratio = push_ratio
        self.recovery_ratio = recovery_ratio
        self.glide_ratio = glide_ratio
        self.step_length = step_length
        self.step_height = step_height
        self._configure_style(style)

    def _configure_style(self, style):
        if style in ["handstand_sync", "front_alt"]:
            self.freeze_legs = ["RL_calf", "RR_calf"]
            self.alternating_legs = [["FL_calf"], ["FR_calf"]]
        elif style == "back_alt":
            self.freeze_legs = ["FL_calf", "FR_calf"]
            self.alternating_legs = [["RL_calf"], ["RR_calf"]]
        elif style == "diagonal_sync":
            self.freeze_legs = []
            self.alternating_legs = [["FL_calf", "RR_calf"], ["FR_calf", "RL_calf"]]
        else:
            self.freeze_legs = []
            self.alternating_legs = [["FL_calf", "RL_calf"], ["FR_calf", "RR_calf"]]

    def set_base_init_feet_pos(self, vx=1.0, yaw=0.0, dt=0.002, yaw_rate=0.0):
        for leg in self.base_feet_pos.keys():
            self.base_feet_pos[leg] += delta_vector(vx=vx, theta=yaw, dt=dt, yaw_rate=yaw_rate)

    def _compute_leg_phase(self, leg_name, t):
        if leg_name in self.freeze_legs:
            return 0.0, False
        period = 2.0 / self.freq
        time_in_period = t % period
        cycle_time = 1.0 / self.freq

        if leg_name in self.alternating_legs[0]:
            return (time_in_period / cycle_time, True) if time_in_period < cycle_time else (1.0, False)
        elif leg_name in self.alternating_legs[1]:
            return ((time_in_period - cycle_time) / cycle_time, True) if time_in_period >= cycle_time else (1.0, False)
        return 0.0, False

    def foot_target(self, leg_name, t, **kwargs):
        phase, should_execute = self._compute_leg_phase(leg_name, t)
        foot = self.base_feet_pos[leg_name].copy()
        if not should_execute:
            return foot

        if phase < self.push_ratio:
            foot[0] -= self.step_length * (phase / self.push_ratio)
        elif phase < self.push_ratio + self.recovery_ratio:
            progress = (phase - self.push_ratio) / self.recovery_ratio
            angle = np.pi * progress
            foot[0] -= self.step_length * (1 - progress)
            foot[2] += self.step_height * np.sin(angle)
        return foot
