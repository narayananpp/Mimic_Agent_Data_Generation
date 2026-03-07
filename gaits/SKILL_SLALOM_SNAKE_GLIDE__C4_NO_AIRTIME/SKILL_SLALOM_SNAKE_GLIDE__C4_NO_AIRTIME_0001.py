from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_SLALOM_SNAKE_GLIDE_MotionGenerator(BaseMotionGenerator):
    """
    Snake-like slalom glide motion with sinusoidal body undulation.
    
    - Constant forward velocity
    - Sinusoidal yaw rate creates lateral path curvature
    - All four feet maintain ground contact throughout cycle
    - Legs adjust laterally and longitudinally in body frame to support curves
    - Phase 0.0-0.25: curve right
    - Phase 0.25-0.5: straighten from right
    - Phase 0.5-0.75: curve left
    - Phase 0.75-1.0: straighten from left
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.vx_forward = 0.6
        self.yaw_amplitude = 1.2
        
        # Leg adjustment amplitudes for body curve support
        self.lateral_shift_amplitude = 0.08
        self.longitudinal_shift_amplitude = 0.05
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base with constant forward velocity and sinusoidal yaw rate.
        
        Yaw rate profile: yaw_rate(phase) = A * sin(2π * phase)
        - phase 0.0: yaw_rate = 0 (neutral)
        - phase 0.125: yaw_rate = +A (peak right turn)
        - phase 0.25: yaw_rate = 0 (neutral)
        - phase 0.5: yaw_rate = 0 (neutral)
        - phase 0.625: yaw_rate = -A (peak left turn)
        - phase 0.75: yaw_rate = 0 (neutral)
        - phase 1.0: yaw_rate = 0 (back to start)
        """
        vx = self.vx_forward
        yaw_rate = self.yaw_amplitude * np.sin(2.0 * np.pi * phase)
        
        self.vel_world = np.array([vx, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with sinusoidal lateral and longitudinal adjustments.
        
        Lateral adjustment (y):
        - Right legs (FR, RR): shift outward during right curve (phase 0-0.25), 
                               inward during left curve (phase 0.5-0.75)
        - Left legs (FL, RL): shift inward during right curve (phase 0-0.25),
                              outward during left curve (phase 0.5-0.75)
        
        Longitudinal adjustment (x):
        - Outer legs trail (negative x shift)
        - Inner legs lead (positive x shift)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if left or right leg
        is_right_leg = leg_name.startswith('FR') or leg_name.startswith('RR')
        
        # Sinusoidal lateral shift profile synchronized with yaw rate
        # sin(2π * phase): +1 at phase=0.25 (right curve peak), -1 at phase=0.75 (left curve peak)
        lateral_phase_signal = np.sin(2.0 * np.pi * phase)
        
        if is_right_leg:
            # Right legs: extend outward (+y) during right curve (positive signal)
            #             tuck inward (-y) during left curve (negative signal)
            lateral_adjustment = self.lateral_shift_amplitude * lateral_phase_signal
            longitudinal_adjustment = -self.longitudinal_shift_amplitude * lateral_phase_signal
        else:
            # Left legs: tuck inward (-y) during right curve (positive signal)
            #            extend outward (+y) during left curve (negative signal)
            lateral_adjustment = -self.lateral_shift_amplitude * lateral_phase_signal
            longitudinal_adjustment = self.longitudinal_shift_amplitude * lateral_phase_signal
        
        foot[0] += longitudinal_adjustment
        foot[1] += lateral_adjustment
        
        return foot