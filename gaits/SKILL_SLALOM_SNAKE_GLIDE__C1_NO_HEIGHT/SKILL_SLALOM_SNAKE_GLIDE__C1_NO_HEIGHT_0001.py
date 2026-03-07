from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SLALOM_SNAKE_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Snake-like slalom glide with continuous quadrupedal support.
    
    - All four feet maintain ground contact throughout the cycle
    - Base executes sinusoidal lateral velocity and yaw rate for slalom path
    - Constant forward velocity ensures steady forward progress
    - Leg positions shift in body frame to accommodate body curvature:
      * Right curve (0-0.25): right legs trail, left legs lead
      * Left curve (0.5-0.75): left legs trail, right legs lead
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Cycle frequency for slalom motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Forward velocity (constant)
        self.vx = 0.5
        
        # Lateral velocity amplitude (creates lateral sinusoidal motion)
        self.vy_amp = 0.3
        
        # Yaw rate amplitude (creates body rotation for slalom)
        self.yaw_rate_amp = 0.8
        
        # Leg repositioning parameters in body frame
        # These define how much legs shift fore-aft to accommodate body curvature
        self.leg_shift_amplitude = 0.08  # Forward/backward shift amplitude
        self.leg_lateral_shift = 0.02    # Small lateral adjustment

    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity combined with
        sinusoidal lateral velocity and yaw rate to create slalom path.
        
        phase 0.0-0.25: curve right (vy > 0, yaw_rate > 0)
        phase 0.25-0.5: transition to left (vy decreasing, yaw_rate decreasing)
        phase 0.5-0.75: curve left (vy < 0, yaw_rate < 0)
        phase 0.75-1.0: transition to right (vy increasing, yaw_rate increasing)
        """
        
        # Constant forward velocity
        vx = self.vx
        
        # Sinusoidal lateral velocity: positive = right, negative = left
        # sin(2π * phase) gives: 0→1→0→-1→0 over phase [0,1]
        vy = self.vy_amp * np.sin(2 * np.pi * phase)
        
        # Sinusoidal yaw rate synchronized with lateral motion
        yaw_rate = self.yaw_rate_amp * np.sin(2 * np.pi * phase)
        
        # Set velocity commands
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
        Compute foot position in body frame with smooth repositioning
        to accommodate body curvature during slalom motion.
        
        During right curve (phase 0-0.25):
          - Right legs (FR, RR): trail backward, shift inward
          - Left legs (FL, RL): lead forward, shift outward
        
        During left curve (phase 0.5-0.75):
          - Left legs (FL, RL): trail backward, shift inward
          - Right legs (FR, RR): lead forward, shift outward
        
        Transitions (0.25-0.5, 0.75-1.0): smooth interpolation through neutral
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Compute forward/backward shift based on phase
        # Right curve phase (0-0.25): sin(2π*phase) > 0
        # Left curve phase (0.5-0.75): sin(2π*phase) < 0
        curve_factor = np.sin(2 * np.pi * phase)
        
        # Fore-aft repositioning
        if is_left_leg:
            # Left legs lead during right curve (positive curve_factor)
            # Left legs trail during left curve (negative curve_factor)
            x_shift = self.leg_shift_amplitude * curve_factor
        elif is_right_leg:
            # Right legs trail during right curve (positive curve_factor)
            # Right legs lead during left curve (negative curve_factor)
            x_shift = -self.leg_shift_amplitude * curve_factor
        else:
            x_shift = 0.0
        
        # Lateral repositioning (smaller magnitude)
        if is_left_leg:
            # Left legs shift outward during right curve, inward during left curve
            y_shift = self.leg_lateral_shift * curve_factor
        elif is_right_leg:
            # Right legs shift inward during right curve, outward during left curve
            y_shift = -self.leg_lateral_shift * curve_factor
        else:
            y_shift = 0.0
        
        # Apply shifts
        foot[0] += x_shift  # Forward/backward in body frame
        foot[1] += y_shift  # Lateral in body frame
        
        # Z remains at ground level (no vertical motion, continuous contact)
        
        return foot