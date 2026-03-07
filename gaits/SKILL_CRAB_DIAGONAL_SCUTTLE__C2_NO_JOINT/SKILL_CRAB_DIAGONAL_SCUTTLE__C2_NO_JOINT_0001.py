from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab-style diagonal scuttle gait with perpendicular body orientation.
    
    Motion phases:
    - [0.0, 0.3]: First scuttle stroke (front legs sweep rearward, rear legs sweep forward)
    - [0.3, 0.5]: Rapid reset (legs return to starting positions)
    - [0.5, 0.8]: Second scuttle stroke (amplified sweep for acceleration)
    - [0.8, 1.0]: Glide stabilization (minimal leg motion, momentum decay)
    
    Body moves diagonally forward-right while oriented perpendicular to travel direction.
    All feet remain in contact throughout the cycle.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scuttle parameters
        self.sweep_amplitude_x = 0.12  # Longitudinal sweep distance
        self.sweep_amplitude_y = 0.04  # Lateral sweep component
        self.stroke_amplification = 1.4  # Second stroke amplitude multiplier
        
        # Base velocity parameters
        self.vx_stroke1 = 0.5  # Forward velocity during first stroke
        self.vy_stroke1 = 0.5  # Lateral velocity during first stroke
        self.vx_stroke2 = 0.7  # Forward velocity during second stroke (amplified)
        self.vy_stroke2 = 0.7  # Lateral velocity during second stroke (amplified)
        self.vx_reset = 0.25   # Forward velocity during reset
        self.vy_reset = 0.25   # Lateral velocity during reset
        self.yaw_rate_maintain = 0.1  # Small yaw rate to maintain perpendicular orientation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on scuttle phase.
        Body moves diagonally forward-right while maintaining perpendicular orientation.
        """
        if phase < 0.3:
            # First scuttle stroke: moderate diagonal velocity
            vx = self.vx_stroke1
            vy = self.vy_stroke1
            yaw_rate = self.yaw_rate_maintain
            
        elif phase < 0.5:
            # Rapid reset: reduced velocity during leg repositioning
            vx = self.vx_reset
            vy = self.vy_reset
            yaw_rate = 0.0
            
        elif phase < 0.8:
            # Second scuttle stroke: amplified diagonal velocity
            vx = self.vx_stroke2
            vy = self.vy_stroke2
            yaw_rate = self.yaw_rate_maintain
            
        else:
            # Glide stabilization: smooth decay to near-zero
            glide_progress = (phase - 0.8) / 0.2
            decay_factor = 1.0 - glide_progress
            vx = self.vx_stroke2 * decay_factor
            vy = self.vy_stroke2 * decay_factor
            yaw_rate = 0.0
        
        self.vel_world = np.array([vx, vy, 0.0])
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
        Compute foot position in body frame for scuttle motion.
        
        Front legs (FL, FR): sweep rearward during stroke, forward during reset
        Rear legs (RL, RR): sweep forward during stroke, rearward during reset
        All feet maintain ground contact throughout.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('FL') or leg_name.startswith('FR')
        is_left = leg_name.startswith('FL') or leg_name.startswith('RL')
        
        # Lateral sweep direction (outward motion during stroke)
        y_sign = 1.0 if is_left else -1.0
        
        if phase < 0.3:
            # First scuttle stroke
            stroke_progress = phase / 0.3
            
            if is_front:
                # Front legs sweep rearward (negative x direction)
                foot[0] += self.sweep_amplitude_x * (0.5 - stroke_progress)
            else:
                # Rear legs sweep forward (positive x direction)
                foot[0] += self.sweep_amplitude_x * (stroke_progress - 0.5)
            
            # Slight outward lateral motion
            foot[1] += y_sign * self.sweep_amplitude_y * np.sin(np.pi * stroke_progress)
            
        elif phase < 0.5:
            # Rapid reset: legs return to starting positions
            reset_progress = (phase - 0.3) / 0.2
            
            if is_front:
                # Front legs return forward
                foot[0] += self.sweep_amplitude_x * (-0.5 + reset_progress)
            else:
                # Rear legs return rearward
                foot[0] += self.sweep_amplitude_x * (0.5 - reset_progress)
            
            # Lateral return motion
            foot[1] += y_sign * self.sweep_amplitude_y * np.sin(np.pi * (1.0 - reset_progress))
            
        elif phase < 0.8:
            # Second scuttle stroke: amplified motion
            stroke_progress = (phase - 0.5) / 0.3
            amplitude_x = self.sweep_amplitude_x * self.stroke_amplification
            amplitude_y = self.sweep_amplitude_y * self.stroke_amplification
            
            if is_front:
                # Front legs sweep rearward with greater amplitude
                foot[0] += amplitude_x * (0.5 - stroke_progress)
            else:
                # Rear legs sweep forward with greater amplitude
                foot[0] += amplitude_x * (stroke_progress - 0.5)
            
            # Amplified lateral motion
            foot[1] += y_sign * amplitude_y * np.sin(np.pi * stroke_progress)
            
        else:
            # Glide stabilization: legs settle into neutral position
            glide_progress = (phase - 0.8) / 0.2
            amplitude_x = self.sweep_amplitude_x * self.stroke_amplification
            amplitude_y = self.sweep_amplitude_y * self.stroke_amplification
            
            # Smooth transition back to neutral stance
            blend_factor = 1.0 - glide_progress
            
            if is_front:
                foot[0] += amplitude_x * (-0.5) * blend_factor
            else:
                foot[0] += amplitude_x * 0.5 * blend_factor
            
            foot[1] += y_sign * amplitude_y * 0.0  # Lateral motion zeroed
        
        return foot