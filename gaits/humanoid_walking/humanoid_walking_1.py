import numpy as np
from gaits.base import BaseMotionGenerator


class HumanoidWalkMotionGenerator(BaseMotionGenerator):
    """
    Bipedal walking motion generator for humanoid robots.
    Inherits from BaseMotionGenerator to handle world-frame integration.
    """

    def __init__(
            self,
            initial_foot_positions_body: dict,
            leg_names: list,
            freq: float = 1.2,  # Slightly faster for stability
            duty_ratio: float = 0.6,
            step_length: float = 0.05,
            step_height: float = 0.05,
            arm_swing: float = 0.2,
    ):
        """
        initial_foot_positions_body should contain:
        'left_foot', 'right_foot', 'left_hand', 'right_hand'
        """
        super().__init__(
            base_init_feet_pos=initial_foot_positions_body,
            freq=freq,
        )

        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        self.arm_swing = arm_swing
        self.leg_names = leg_names

        # Bipedal phase offsets (feet are 180 deg apart)
        self._phase_offsets = {
            "right_foot": 0.0,
            "left_foot": 0.5,
            "right_hand": 0.5,  # Contralateral swing
            "left_hand": 0.0,
        }

    # ---------------------------------------------------------
    # Trajectory Math
    # ---------------------------------------------------------
    def _foot_step(self, nominal_pos: np.ndarray, leg_phase: float) -> np.ndarray:
        """Computes the XYZ trajectory of a foot in body frame."""
        pos = nominal_pos.copy()

        if leg_phase < self.duty:
            # Stance: foot moves backward relative to body
            # 0 to 1 progress within stance
            p_stance = leg_phase / self.duty
            pos[0] -= self.step_length * (p_stance - 0.5)
        else:
            # Swing: foot moves forward and lifts
            # 0 to 1 progress within swing
            p_swing = (leg_phase - self.duty) / (1.0 - self.duty)
            pos[0] += self.step_length * (p_swing - 0.5)
            pos[2] += self.step_height * np.sin(np.pi * p_swing)

        return pos

    def _hand_swing(self, nominal_pos: np.ndarray, leg_phase: float) -> np.ndarray:
        """Computes the XYZ trajectory of a hand in body frame."""
        pos = nominal_pos.copy()
        # Simple sinusoidal swing
        swing = self.arm_swing * np.sin(2.0 * np.pi * leg_phase)
        pos[0] += swing
        pos[2] += 0.02 * np.cos(2.0 * np.pi * leg_phase)  # Slight vertical arc
        return pos

    # ---------------------------------------------------------
    # BaseMotionGenerator Overrides
    # ---------------------------------------------------------
    def compute_foot_position_body_frame(self, leg_name: str, phase: float) -> np.ndarray:
        """Mapping canonical names to specific trajectory functions."""
        nominal = self.base_init_feet_pos[leg_name]
        leg_phase = (phase + self._phase_offsets.get(leg_name, 0.0)) % 1.0

        if "hand" in leg_name:
            return self._hand_swing(nominal, leg_phase)
        return self._foot_step(nominal, leg_phase)

    def step(self, dt):
        """
        Overriding step to inject joint overrides for joints not controlled by IK.
        """
        # Call base step to handle root integration and IK target generation
        state = super().step(dt)

        # Add shoulder/hip-yaw stabilization overrides
        phase = state["phase"]
        swing_val = 0.3 * np.sin(2.0 * np.pi * phase)

        state["joint_overrides"] = {
            # Direct pose control for non-locomotion joints
            "right_shoulder_x": 1.57,  # Hold arms out slightly
            "left_shoulder_x": -1.57,
            "right_shoulder_z": swing_val,
            "left_shoulder_z": swing_val,
            "right_hip_z": 0.0,  # Lock hip yaw to prevent spaghetti legs
            "left_hip_z": 0.0,
        }

        return state