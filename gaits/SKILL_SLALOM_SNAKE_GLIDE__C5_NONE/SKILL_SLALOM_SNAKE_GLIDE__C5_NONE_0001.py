from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SLALOM_SNAKE_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Snake-like slalom glide motion.
    
    - Robot glides forward while tracing a sinusoidal lateral path
    - All four feet maintain continuous ground contact
    - Legs reposition smoothly in body frame to support body curvature
    - Right curve (phase 0-0.25): left legs lead, right legs trail
    - Left curve (phase 0.5-0.75): right legs lead, left legs trail
    - Neutral transitions (0.25-0.5, 0.75-1.0): legs shift positions smoothly
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Slower frequency for smooth slalom motion
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters
        self.vx_forward = 0.6  # Constant forward glide speed
        self.vy_amp = 0.4  # Amplitude of lateral drift
        self.yaw_rate_amp = 0.8  # Amplitude of yaw rotation
        
        # Leg repositioning parameters
        self.leg_shift_amp = 0.15  # Amplitude of forward/backward leg shift in body frame
        
    def update_base_motion(self, phase, dt):
        """
        Update base using constant forward velocity with sinusoidal lateral drift and yaw.
        
        Lateral velocity (vy):
          - Peak right at phase ~0.125: vy > 0
          - Zero at phase 0.25
          - Peak left at phase ~0.625: vy < 0
          - Zero at phase 0.75
          Formula: vy = vy_amp * sin(2π * phase)
          
        Yaw rate:
          - Positive during right curve (phase 0-0.25)
          - Negative during left curve (phase 0.5-0.75)
          - Phase-shifted relative to lateral velocity for coordinated curves
          Formula: yaw_rate = yaw_rate_amp * cos(2π * phase)
        """
        
        # Constant forward velocity
        vx = self.vx_forward
        
        # Sinusoidal lateral velocity: positive = right, negative = left
        # sin(2π * phase): 0 at phase=0, peak at 0.25, 0 at 0.5, trough at 0.75, 0 at 1.0
        # Shift to get peak at 0.125: use sin(2π * (phase - 0.125)) = -cos(2π * phase)
        vy = -self.vy_amp * np.cos(2 * np.pi * phase)
        
        # Zero vertical velocity
        vz = 0.0
        
        # Sinusoidal yaw rate: positive = CCW (right curve), negative = CW (left curve)
        # cos(2π * phase): peak at phase=0, 0 at 0.25, trough at 0.5, 0 at 0.75, peak at 1.0
        # This gives positive yaw during right curve and negative during left curve
        yaw_rate = self.yaw_rate_amp * np.cos(2 * np.pi * phase)
        
        # Set velocity commands in world frame
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
        Compute foot position in body frame with smooth sinusoidal repositioning.
        
        During right curve (phase 0-0.25):
          - Left legs (FL, RL): shift forward (lead)
          - Right legs (FR, RR): shift backward (trail)
          
        During left curve (phase 0.5-0.75):
          - Right legs (FR, RR): shift forward (lead)
          - Left legs (FL, RL): shift backward (trail)
          
        Transitions (0.25-0.5, 0.75-1.0): smooth sinusoidal blending
        """
        
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if this is a left or right leg
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Compute forward/backward shift using sinusoidal function
        # For left legs: shift forward during right curve (phase 0-0.25), backward during left curve (0.5-0.75)
        # For right legs: shift backward during right curve (phase 0-0.25), forward during left curve (0.5-0.75)
        #
        # Use sin(2π * phase):
        #   phase 0: 0
        #   phase 0.25: 1 (peak positive)
        #   phase 0.5: 0
        #   phase 0.75: -1 (peak negative)
        #   phase 1.0: 0
        
        shift_signal = np.sin(2 * np.pi * phase)
        
        if is_left:
            # Left legs: positive shift when signal is positive (right curve)
            x_shift = self.leg_shift_amp * shift_signal
        else:
            # Right legs: negative shift when signal is positive (right curve)
            x_shift = -self.leg_shift_amp * shift_signal
        
        # Apply forward/backward shift
        foot[0] += x_shift
        
        # All feet remain on ground (z = base position maintained)
        # No vertical adjustment needed for continuous ground contact
        
        return foot