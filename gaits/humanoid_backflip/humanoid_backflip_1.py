import numpy as np
from gaits.base import BaseMotionGenerator


class HumanoidBackflipMotionGenerator(BaseMotionGenerator):
    """
    Backflip motion generator for humanoid robots.

    Phase breakdown (0 → 1):
        0.00 - 0.15  : Crouch     — bend knees, lean slightly forward
        0.15 - 0.30  : Launch     — explosive hip/knee extension, arms swing up
        0.30 - 0.70  : Flight     — full tuck, arms overhead, body rotates backward
        0.70 - 0.85  : Open       — extend legs for landing
        0.85 - 1.00  : Land       — absorb impact, return to stand

    Root rotation:
        The backward somersault is driven by setting root_quat directly
        via update_base_motion — full 2π rotation around the X axis.
    """

    def __init__(
            self,
            initial_foot_positions_body: dict,
            leg_names: list,
            freq: float = 0.4,      # slow — one backflip per 2.5 seconds
            jump_height: float = 0.5,  # peak height above ground (m)
    ):
        super().__init__(
            base_init_feet_pos=initial_foot_positions_body,
            freq=freq,
        )

        self.leg_names    = leg_names
        self.jump_height  = jump_height

        # store initial pelvis height for trajectory reference
        self._init_z = None  # set on first step

        # phase boundaries
        self.P_CROUCH_END  = 0.15
        self.P_LAUNCH_END  = 0.30
        self.P_FLIGHT_END  = 0.70
        self.P_OPEN_END    = 0.85
        # 0.85 → 1.0 = landing

        self._phase_offsets = {
            "right_foot": 0.0,
            "left_foot":  0.0,   # both feet move together for backflip
            "right_hand": 0.0,
            "left_hand":  0.0,
        }

    # ---------------------------------------------------------
    # Root motion — drive the backward rotation
    # ---------------------------------------------------------
    def update_base_motion(self, phase, dt):
        """
        Override base integration.
        - No forward/lateral translation
        - Vertical trajectory: crouch → jump → peak → land
        - Backward rotation: 0 → 2π over flight phase
        """
        from utils.math_utils import integrate_pose_world_frame

        if self._init_z is None:
            self._init_z = self.root_pos[2]

        z = self._init_z  # default: stay at initial height

        if phase < self.P_CROUCH_END:
            # crouch — lower center of mass
            p = phase / self.P_CROUCH_END
            z = self._init_z - 0.15 * np.sin(np.pi * p / 2.0)

        elif phase < self.P_LAUNCH_END:
            # launch — rise from crouch to peak
            p = (phase - self.P_CROUCH_END) / (self.P_LAUNCH_END - self.P_CROUCH_END)
            z = (self._init_z - 0.15) + (self.jump_height + 0.15) * p

        elif phase < self.P_FLIGHT_END:
            # flight — parabolic arc, full rotation
            p = (phase - self.P_LAUNCH_END) / (self.P_FLIGHT_END - self.P_LAUNCH_END)
            # parabola: peak at p=0.5
            z = self._init_z + self.jump_height * (1.0 - (2 * p - 1) ** 2)

        elif phase < self.P_OPEN_END:
            # open — descending
            p = (phase - self.P_FLIGHT_END) / (self.P_OPEN_END - self.P_FLIGHT_END)
            z = self._init_z + self.jump_height * (1.0 - p)

        else:
            # land — absorb
            p = (phase - self.P_OPEN_END) / (1.0 - self.P_OPEN_END)
            z = self._init_z - 0.1 * np.sin(np.pi * p / 2.0)

        # backward rotation (around X axis) during flight phase
        roll_angle = 0.0
        if self.P_LAUNCH_END <= phase < self.P_OPEN_END:
            p_rot = (phase - self.P_LAUNCH_END) / (self.P_OPEN_END - self.P_LAUNCH_END)
            roll_angle = -2.0 * np.pi * p_rot  # full backward rotation

        # build quaternion from roll angle (rotation around X)
        half = roll_angle / 2.0
        self.root_quat = np.array([
            np.cos(half),  # w
            np.sin(half),  # x
            0.0,           # y
            0.0,           # z
        ])
        self.root_quat /= np.linalg.norm(self.root_quat)

        # update position — no forward motion, only vertical
        self.root_pos[2] = z

    # ---------------------------------------------------------
    # Foot trajectory — both feet together
    # ---------------------------------------------------------
    def compute_foot_position_body_frame(self, leg_name: str, phase: float) -> np.ndarray:
        pos = self.base_init_feet_pos[leg_name].copy()

        if "hand" in leg_name:
            return self._hand_trajectory(pos, phase)
        return self._foot_trajectory(pos, phase)

    def _foot_trajectory(self, pos: np.ndarray, phase: float) -> np.ndarray:
        """
        Both feet together:
        - Crouch:  feet stay planted, knees bend (handled via joint overrides)
        - Launch:  feet push off (stay at ground relative to body)
        - Flight:  tuck — pull feet up toward body
        - Open:    extend feet back down
        - Land:    feet plant, absorb
        """
        if phase < self.P_CROUCH_END:
            pass  # feet stay at nominal

        elif phase < self.P_LAUNCH_END:
            pass  # feet push off, stay nominal

        elif phase < self.P_FLIGHT_END:
            # tuck — pull feet up toward pelvis
            p = (phase - self.P_LAUNCH_END) / (self.P_FLIGHT_END - self.P_LAUNCH_END)
            tuck = np.sin(np.pi * p)  # 0 → 1 → 0
            pos[2] += 0.4 * tuck      # pull feet up

        elif phase < self.P_OPEN_END:
            # extend legs for landing
            p = (phase - self.P_FLIGHT_END) / (self.P_OPEN_END - self.P_FLIGHT_END)
            pos[2] -= 0.1 * p         # push feet back down

        return pos

    def _hand_trajectory(self, pos: np.ndarray, phase: float) -> np.ndarray:
        """
        Arms:
        - Crouch:  arms hang at sides
        - Launch:  arms swing up aggressively (adds angular momentum)
        - Flight:  arms overhead in tuck
        - Open:    arms come forward/down
        - Land:    arms return to sides
        """
        if phase < self.P_CROUCH_END:
            pass  # arms at sides

        elif phase < self.P_LAUNCH_END:
            # swing arms up
            p = (phase - self.P_CROUCH_END) / (self.P_LAUNCH_END - self.P_CROUCH_END)
            pos[2] += 0.5 * p   # arms swing up

        elif phase < self.P_FLIGHT_END:
            # arms overhead
            pos[2] += 0.5

        elif phase < self.P_OPEN_END:
            # arms come down
            p = (phase - self.P_FLIGHT_END) / (self.P_OPEN_END - self.P_FLIGHT_END)
            pos[2] += 0.5 * (1.0 - p)

        return pos

    # ---------------------------------------------------------
    # Step — inject joint overrides per phase
    # ---------------------------------------------------------
    def step(self, dt):
        state = super().step(dt)
        phase = state["phase"]

        overrides = {}

        if phase < self.P_CROUCH_END:
            # ----- CROUCH -----
            p = phase / self.P_CROUCH_END
            crouch = np.sin(np.pi * p / 2.0)  # 0 → 1

            overrides = {
                "right_shoulder_x":  1.57,
                "left_shoulder_x":  -1.57,
                "right_shoulder_z":  0.0,
                "left_shoulder_z":   0.0,
                "abdomen_y":         0.1 * crouch,   # slight forward lean
                "right_hip_z":       0.0,
                "left_hip_z":        0.0,
                "right_knee":        0.8 * crouch,   # bend knees
                "left_knee":         0.8 * crouch,
            }

        elif phase < self.P_LAUNCH_END:
            # ----- LAUNCH -----
            p = (phase - self.P_CROUCH_END) / (self.P_LAUNCH_END - self.P_CROUCH_END)
            extend = p  # 0 → 1

            overrides = {
                # arms swing up aggressively
                "right_shoulder_x":  1.57 - 2.5 * extend,
                "left_shoulder_x":  -1.57 + 2.5 * extend,
                "right_shoulder_z":  0.0,
                "left_shoulder_z":   0.0,
                "abdomen_y":         0.1 * (1.0 - extend),
                "right_hip_z":       0.0,
                "left_hip_z":        0.0,
                "right_knee":        0.8 * (1.0 - extend),  # extend knees
                "left_knee":         0.8 * (1.0 - extend),
            }

        elif phase < self.P_FLIGHT_END:
            # ----- FLIGHT / TUCK -----
            p = (phase - self.P_LAUNCH_END) / (self.P_FLIGHT_END - self.P_LAUNCH_END)
            tuck = np.sin(np.pi * p)   # 0 → 1 → 0

            overrides = {
                # arms overhead
                "right_shoulder_x": -1.0,
                "left_shoulder_x":   1.0,
                "right_shoulder_z":  0.0,
                "left_shoulder_z":   0.0,
                # tuck — bend knees and hips tightly
                "right_knee":        2.0 * tuck,
                "left_knee":         2.0 * tuck,
                "right_hip_y":       1.5 * tuck,
                "left_hip_y":        1.5 * tuck,
                "abdomen_y":        -0.3 * tuck,   # curl body
                "right_hip_z":       0.0,
                "left_hip_z":        0.0,
            }

        elif phase < self.P_OPEN_END:
            # ----- OPEN -----
            p = (phase - self.P_FLIGHT_END) / (self.P_OPEN_END - self.P_FLIGHT_END)
            open_amt = p  # 0 → 1

            overrides = {
                # arms come forward/down
                "right_shoulder_x":  -1.0 + 2.57 * open_amt,
                "left_shoulder_x":    1.0 - 2.57 * open_amt,
                "right_shoulder_z":   0.0,
                "left_shoulder_z":    0.0,
                # extend for landing
                "right_knee":         2.0 * (1.0 - open_amt),
                "left_knee":          2.0 * (1.0 - open_amt),
                "right_hip_y":        1.5 * (1.0 - open_amt),
                "left_hip_y":         1.5 * (1.0 - open_amt),
                "abdomen_y":         -0.3 * (1.0 - open_amt),
                "right_hip_z":        0.0,
                "left_hip_z":         0.0,
            }

        else:
            # ----- LAND -----
            p = (phase - self.P_OPEN_END) / (1.0 - self.P_OPEN_END)
            absorb = np.sin(np.pi * p / 2.0)  # 0 → 1

            overrides = {
                "right_shoulder_x":   1.57,
                "left_shoulder_x":   -1.57,
                "right_shoulder_z":   0.0,
                "left_shoulder_z":    0.0,
                "right_knee":         0.5 * absorb,   # absorb landing
                "left_knee":          0.5 * absorb,
                "right_hip_y":        0.3 * absorb,
                "left_hip_y":         0.3 * absorb,
                "abdomen_y":          0.1 * absorb,
                "right_hip_z":        0.0,
                "left_hip_z":         0.0,
            }

        state["joint_overrides"] = overrides
        return state