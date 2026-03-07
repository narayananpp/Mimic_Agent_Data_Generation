import numpy as np
from utils.math_utils import *
from gaits.base import BaseMotionGenerator


class WalkingMotionGenerator(BaseMotionGenerator):

    def __init__(self, initial_foot_positions_body, leg_names,
                 freq=1.0, duty_ratio=0.75, step_length=0.1, step_height=0.05):
        super().__init__(initial_foot_positions_body, freq)

        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height

        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Diagonal trot phase offsets
        self.phase_offsets = {
            leg_names[0]: 0.0,   # FL
            leg_names[1]: 0.5,   # FR
            leg_names[2]: 0.5,   # RL
            leg_names[3]: 0.0,   # RR
        }

    def compute_foot_position_body_frame(self, leg_name, phase):
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()

        if leg_phase < self.duty:
            progress = leg_phase / self.duty
            foot[0] -= self.step_length * (progress - 0.5)
        else:
            progress = (leg_phase - self.duty) / (1.0 - self.duty)
            foot[0] += self.step_length * (progress - 0.5)
            foot[2] += self.step_height * np.sin(np.pi * progress)

        return foot