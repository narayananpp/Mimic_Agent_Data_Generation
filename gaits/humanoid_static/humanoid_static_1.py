# gaits/humanoid_static/humanoid_static_1.py
"""
Static motion generator for humanoid robot.

Keeps all end effectors (feet + hands) at their initial positions
in body frame — no locomotion, just a stable standing/static pose.

End effectors:
    - right_foot, left_foot  (contact with ground)
    - right_hand, left_hand  (upper body, optional)
"""

import numpy as np
from gaits.base import BaseMotionGenerator


class HumanoidStaticMotionGenerator(BaseMotionGenerator):
    """
    Static pose generator for humanoid.

    All end effectors held at initial body-frame positions.
    Root pose is completely frozen — no integration, no drift.

    Suitable for:
        - Recording reference standing pose
        - Baseline before locomotion skills
        - Handstand / still poses
    """

    def __init__(
        self,
        initial_foot_positions_body: dict,
        leg_names: list,
        freq: float = 1.0,
        style: str = "stand",
    ):
        super().__init__(
            base_init_feet_pos=initial_foot_positions_body,
            freq=freq,
        )

        self.style = style
        self.leg_names = leg_names

        print(f"[HumanoidStaticMotionGenerator] style={style}")
        print(f"[HumanoidStaticMotionGenerator] end effectors: {leg_names}")
        for name, pos in initial_foot_positions_body.items():
            print(f"  {name}: {np.round(pos, 4)}")

    # ------------------------------------------------------------------
    # ROOT MOTION — frozen, no integration
    # ------------------------------------------------------------------
    def update_base_motion(self, phase, dt):
        """
        Override base class — do NOT integrate velocity.
        Root stays exactly where it was reset to.
        """
        pass  # ← root_pos and root_quat never change

    # ------------------------------------------------------------------
    # END EFFECTOR TARGETS — frozen at initial positions
    # ------------------------------------------------------------------
    def compute_foot_position_body_frame(self, leg_name: str, phase: float) -> np.ndarray:
        """
        Return the initial body-frame position for this end effector.
        Static — never moves.
        """
        return self.base_feet_pos[leg_name].copy()