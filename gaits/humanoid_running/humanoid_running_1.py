import numpy as np
from gaits.base import BaseMotionGenerator


class HumanoidRunMotionGenerator(BaseMotionGenerator):
    """
    Bipedal running motion generator for humanoid robots.

    Key differences from walking:
    - Lower duty ratio (less stance time, more flight phase)
    - Longer step length
    - Higher step height (more aggressive lift)
    - Faster frequency
    - More pronounced arm swing
    - Slight forward body lean via abdomen override
    """

    def __init__(
            self,
            initial_foot_positions_body: dict,
            leg_names: list,
            freq: float = 2.0,          # faster than walk (1.2 Hz)
            duty_ratio: float = 0.4,    # less stance than walk (0.6)
            step_length: float = 0.12,  # longer stride than walk (0.05)
            step_height: float = 0.10,  # higher lift than walk (0.05)
            arm_swing: float = 0.35,    # more aggressive than walk (0.2)
    ):
        super().__init__(
            base_init_feet_pos=initial_foot_positions_body,
            freq=freq,
        )

        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        self.arm_swing = arm_swing
        self.leg_names = leg_names

        # Bipedal phase offsets — same as walk, feet 180° apart
        self._phase_offsets = {
            "right_foot": 0.0,
            "left_foot":  0.5,
            "right_hand": 0.5,   # contralateral swing
            "left_hand":  0.0,
        }

    # ---------------------------------------------------------
    # Trajectory Math
    # ---------------------------------------------------------
    def _foot_step(self, nominal_pos: np.ndarray, leg_phase: float) -> np.ndarray:
        """
        Running foot trajectory — more aggressive than walk.
        Stance is shorter, swing is higher and faster.
        """
        pos = nominal_pos.copy()

        if leg_phase < self.duty:
            # Stance: foot pushes back hard
            p_stance = leg_phase / self.duty
            pos[0] -= self.step_length * (p_stance - 0.5)
        else:
            # Swing: foot lifts high and swings far forward
            p_swing = (leg_phase - self.duty) / (1.0 - self.duty)
            pos[0] += self.step_length * (p_swing - 0.5)
            # higher parabolic lift for running
            pos[2] += self.step_height * np.sin(np.pi * p_swing)

        return pos

    def _hand_swing(self, nominal_pos: np.ndarray, leg_phase: float) -> np.ndarray:
        """
        Running arm swing — more pronounced, slightly bent elbows.
        """
        pos = nominal_pos.copy()
        swing = self.arm_swing * np.sin(2.0 * np.pi * leg_phase)
        pos[0] += swing
        # slight vertical arc during running
        pos[2] += 0.04 * np.cos(2.0 * np.pi * leg_phase)
        return pos

    # ---------------------------------------------------------
    # BaseMotionGenerator Overrides
    # ---------------------------------------------------------
    def compute_foot_position_body_frame(self, leg_name: str, phase: float) -> np.ndarray:
        nominal = self.base_init_feet_pos[leg_name]
        leg_phase = (phase + self._phase_offsets.get(leg_name, 0.0)) % 1.0

        if "hand" in leg_name:
            return self._hand_swing(nominal, leg_phase)
        return self._foot_step(nominal, leg_phase)

    def step(self, dt):
        state = super().step(dt)

        phase = state["phase"]

        # arm swing — contralateral (right arm opposite to right leg)
        right_arm_phase = (phase + 0.5) % 1.0
        swing_val = self.arm_swing * np.sin(2.0 * np.pi * right_arm_phase)

        # forward lean — tilt torso forward for running posture
        # abdomen_y positive = lean forward
        forward_lean = 0.15   # ~8.6° forward lean

        # elbow bend — arms bent at ~90° during running
        right_elbow = 1.2 + 0.2 * np.sin(2.0 * np.pi * right_arm_phase)
        left_elbow  = 1.2 - 0.2 * np.sin(2.0 * np.pi * right_arm_phase)
        right_elbow = np.clip(right_elbow, 0.0, 2.7925)
        left_elbow  = np.clip(left_elbow,  0.0, 2.7925)

        state["joint_overrides"] = {
            # torso lean forward
            "abdomen_y":        forward_lean,
            "abdomen_x":        0.0,
            "abdomen_z":        0.0,

            # arms down and bent
            "right_shoulder_x":  1.57,
            "left_shoulder_x":  -1.57,

            # arm swing forward/backward
            "right_shoulder_z":  swing_val,
            "left_shoulder_z":  swing_val,

            # elbow bent for running form
            "right_elbow":       right_elbow,
            "left_elbow":        -left_elbow,

            # lock hip yaw to prevent leg splay
            "right_hip_z":       0.0,
            "left_hip_z":        0.0,
        }

        return state