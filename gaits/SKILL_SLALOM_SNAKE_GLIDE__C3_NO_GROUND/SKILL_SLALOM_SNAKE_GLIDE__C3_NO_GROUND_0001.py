from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_SLALOM_SNAKE_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Slalom snake glide skill: continuous forward gliding motion with sinusoidal
    lateral deviation, creating a snake-like slithering movement.
    
    - All four feet maintain continuous ground contact (no swing phase)
    - Base moves forward at constant velocity while oscillating laterally
    - Yaw rate synchronizes with lateral motion to create curved path
    - Legs adjust lateral positions asymmetrically to support body curvature
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # One complete sinusoidal cycle per skill period
        
        # Base foot positions (neutral stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx_forward = 0.6  # Constant forward velocity
        self.vy_amplitude = 0.4  # Lateral velocity amplitude (controls slalom width)
        self.yaw_rate_amplitude = 0.8  # Yaw rate amplitude (synchronized with lateral motion)
        
        # Leg lateral displacement amplitude (creates body curvature)
        self.leg_lateral_amplitude = 0.12  # How far legs extend laterally from neutral
        self.leg_longitudinal_amplitude = 0.08  # Forward/backward adjustment during curve
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity, sinusoidal lateral velocity,
        and synchronized yaw rate to create smooth slalom path.
        """
        # Constant forward velocity
        vx = self.vx_forward
        
        # Sinusoidal lateral velocity: zero at 0, max right at 0.25, zero at 0.5, max left at 0.75, zero at 1.0
        vy = self.vy_amplitude * np.sin(2 * np.pi * phase)
        
        # Zero vertical velocity (maintain constant height)
        vz = 0.0
        
        # Yaw rate synchronized with lateral motion (derivative of lateral motion phase)
        # Leads slightly into the curve direction
        yaw_rate = self.yaw_rate_amplitude * np.cos(2 * np.pi * phase)
        
        # Set velocities in world frame
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame with continuous ground contact.
        
        Legs adjust laterally and longitudinally to create body curvature:
        - During right curve (phase 0-0.25): right legs extend right, left legs move medial
        - During left curve (phase 0.5-0.75): left legs extend left, right legs move medial
        - Transitions (0.25-0.5, 0.75-1.0): legs return to neutral positions
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if leg is on left or right side
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Lateral displacement: sinusoidal modulation creates body curvature
        # Left legs: positive displacement when phase favors left curve
        # Right legs: positive displacement when phase favors right curve
        if is_left:
            # Left legs extend leftward (negative y) during left curve (phase ~0.5-0.75)
            lateral_offset = -self.leg_lateral_amplitude * np.sin(2 * np.pi * phase)
        else:
            # Right legs extend rightward (positive y) during right curve (phase ~0.0-0.25)
            lateral_offset = self.leg_lateral_amplitude * np.sin(2 * np.pi * phase)
        
        foot[1] += lateral_offset
        
        # Longitudinal displacement: legs on outside of curve trail slightly
        # This creates more natural body curvature
        if is_left:
            # Left legs trail during left curve (when sin is negative for left curve)
            longitudinal_offset = -self.leg_longitudinal_amplitude * np.sin(2 * np.pi * phase)
        else:
            # Right legs trail during right curve (when sin is positive for right curve)
            longitudinal_offset = self.leg_longitudinal_amplitude * np.sin(2 * np.pi * phase)
        
        # Front legs lead, rear legs trail
        if is_front:
            foot[0] += longitudinal_offset * 0.5
        else:
            foot[0] -= longitudinal_offset * 0.5
        
        # Maintain ground contact (z = constant, no vertical motion)
        # foot[2] remains at base position
        
        return foot