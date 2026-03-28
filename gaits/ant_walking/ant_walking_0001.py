import numpy as np
from gaits.base import BaseMotionGenerator

class AntWalkMotionGenerator(BaseMotionGenerator):
    """
    Quadrupedal walking motion generator for the Ant robot.
    Now correctly handles canonical names (FL, FR, RL, RR).
    """

    def __init__(
        self,
        initial_foot_positions_body: dict,
        leg_names: list,
        freq: float = 1.0,
        duty_ratio: float = 0.75,
        step_length: float = 0.08,
        step_height: float = 0.05,
    ):
        super().__init__(
            base_init_feet_pos=initial_foot_positions_body,
            freq=freq,
        )

        self.duty = duty_ratio
        self.step_length = step_length
        self.step_height = step_height
        self.leg_names = leg_names

        # Phase offsets for diagonal trot
        self._phase_offsets = {
            "FL": 0.0, "RR": 0.0,
            "FR": 0.5, "RL": 0.5,
        }

        # Step directions (diagonal splay)
        self._step_dirs = {
            "FR": np.array([ 1.0, -1.0, 0.0]) / np.sqrt(2),
            "FL": np.array([ 1.0,  1.0, 0.0]) / np.sqrt(2),
            "RR": np.array([-1.0, -1.0, 0.0]) / np.sqrt(2),
            "RL": np.array([-1.0,  1.0, 0.0]) / np.sqrt(2),
        }

    def _get_logic_key(self, leg_name: str) -> str:
        """Helper to find FL/FR/RL/RR in a string like 'FR_ankle'."""
        for key in self._phase_offsets.keys():
            if key in leg_name:
                return key
        raise KeyError(f"Leg name {leg_name} does not contain canonical FL/FR/RL/RR")

    def _foot_step(self, nominal_pos: np.ndarray, step_dir: np.ndarray,
                   leg_phase: float) -> np.ndarray:
        pos = nominal_pos.copy()

        # Define the 'Target Body Height' (how high the torso is above the floor)
        # 0.25m is a standard standing height for the Ant.
        body_height = 0.25
        floor_z = -body_height

        if leg_phase < self.duty:
            # ---- STANCE: Foot is pinned to the floor ----
            progress = leg_phase / self.duty
            # Sweep backward
            pos += step_dir * self.step_length * (0.5 - progress)
            # Force Z to be exactly at floor level relative to body
            pos[2] = floor_z
        else:
            # ---- SWING: Foot lifts off the floor ----
            progress = (leg_phase - self.duty) / (1.0 - self.duty)
            # Sweep forward
            pos += step_dir * self.step_length * (progress - 0.5)
            # Lift sinusoidally ABOVE the floor_z
            pos[2] = floor_z + (self.step_height * np.sin(np.pi * progress))

        return pos

    def compute_foot_position_body_frame(self, leg_name: str, phase: float) -> np.ndarray:
        key = self._get_logic_key(leg_name)
        nominal = self.base_init_feet_pos[leg_name]
        step_dir = self._step_dirs[key]
        leg_phase = (phase + self._phase_offsets[key]) % 1.0
        return self._foot_step(nominal, step_dir, leg_phase)

    def step(self, dt):
        # 1. Run the base logic (calculates IK and world positions)
        state = super().step(dt)

        # 2. Force the Root Z-height to 0.25 so the body doesn't drift
        state["root_pos"][2] = 0.25

        # 3. Add joint overrides (matches your Humanoid reference)
        # This ensures the dictionary exists so controller.py line 173 doesn't fail
        state["joint_overrides"] = {}

        return state