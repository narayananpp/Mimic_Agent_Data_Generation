from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SLALOM_SNAKE_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Slalom snake glide gait with sinusoidal lateral body curvature.

    - All four feet remain in ground contact throughout (quasi-static gliding).
    - Base linear velocity: constant forward vx, sinusoidal lateral vy.
    - Base angular velocity: sinusoidal yaw rate, creating S-curve path.
    - Foot trajectories in BODY frame continuously adjust to support body curvature.
    """

    def __init__(self, initial_foot_positions_body, leg_names):

        self.leg_names = leg_names
        self.freq = 0.5  # Slalom cycle frequency (Hz)

        # Base foot positions (nominal stance in body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}

        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Velocity parameters
        self.vx_constant = 0.5  # Constant forward velocity (m/s)
        self.vy_amplitude = 0.3  # Lateral velocity amplitude (m/s)
        self.yaw_rate_amplitude = 1.2  # Yaw rate amplitude (rad/s)

        # Foot modulation parameters (body frame adjustments)
        self.lateral_extension = 0.08  # Lateral foot extension during curves (m)
        self.longitudinal_shift = 0.06  # Forward/backward foot shift during curves (m)

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity and sinusoidal lateral/yaw velocities.
        
        Phase mapping:
        - phase 0.0 → 0.25: right curve (vy > 0, yaw_rate > 0)
        - phase 0.25 → 0.5: transition to left (vy crosses zero, yaw_rate crosses zero)
        - phase 0.5 → 0.75: left curve (vy < 0, yaw_rate < 0)
        - phase 0.75 → 1.0: transition to right (return to zero)
        """
        
        # Constant forward velocity
        vx = self.vx_constant
        
        # Sinusoidal lateral velocity
        # Peak at phase 0.125 (right), zero at 0.375, peak at 0.625 (left), zero at 0.875
        # Use sin(2π * (phase - 0.25)) to get: 0→0.25 positive, 0.5→0.75 negative
        vy = self.vy_amplitude * np.sin(2 * np.pi * (phase - 0.25))
        
        # Sinusoidal yaw rate (same phase as lateral velocity for coordinated curvature)
        yaw_rate = self.yaw_rate_amplitude * np.sin(2 * np.pi * (phase - 0.25))
        
        # Set world-frame velocities
        self.vel_world = np.array([vx, vy, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])

        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with continuous adjustments for body curvature.
        
        Coordination:
        - Right curve (phase 0→0.25): 
            - Right legs (FR, RR): extend laterally (+y), trail (-x)
            - Left legs (FL, RL): tuck inward (-y), lead (+x)
        - Left curve (phase 0.5→0.75):
            - Left legs (FL, RL): extend laterally (+y), trail (-x)
            - Right legs (FR, RR): tuck inward (-y), lead (+x)
        - Transitions (phase 0.25→0.5, 0.75→1.0): smooth sinusoidal interpolation
        """
        
        # Start with nominal foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left_side = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_side = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute lateral and longitudinal modulation using sinusoids
        # For right curve (phase near 0.125): modulation positive
        # For left curve (phase near 0.625): modulation negative
        curve_modulation = np.sin(2 * np.pi * (phase - 0.25))
        
        # Lateral adjustment (y direction in body frame)
        if is_right_side:
            # Right legs: extend outward (+y) during right curve, tuck inward (-y) during left curve
            lateral_adjustment = self.lateral_extension * curve_modulation
        else:  # is_left_side
            # Left legs: tuck inward (-y) during right curve, extend outward (+y) during left curve
            lateral_adjustment = -self.lateral_extension * curve_modulation
        
        # Longitudinal adjustment (x direction in body frame)
        if is_right_side:
            # Right legs: trail (backward, -x) during right curve, lead (forward, +x) during left curve
            longitudinal_adjustment = -self.longitudinal_shift * curve_modulation
        else:  # is_left_side
            # Left legs: lead (forward, +x) during right curve, trail (backward, -x) during left curve
            longitudinal_adjustment = self.longitudinal_shift * curve_modulation
        
        # Apply adjustments
        foot[0] += longitudinal_adjustment  # x (forward/backward in body frame)
        foot[1] += lateral_adjustment        # y (lateral in body frame)
        # foot[2] remains unchanged (no vertical motion, z = 0 for ground contact)
        
        return foot