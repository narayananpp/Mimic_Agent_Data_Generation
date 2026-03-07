from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab diagonal scuttle gait with constant sideways body orientation.
    
    Motion consists of:
    - Two scuttle strokes (front legs sweep backward, rear legs sweep forward)
    - Rapid reset phase with all legs airborne
    - Glide stabilization phase
    - Body maintains ~90° yaw throughout via zero yaw rate
    - Diagonal motion from simultaneous forward and lateral velocities
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scuttle motion parameters - reduced to respect joint limits
        self.first_stroke_amplitude = 0.07  # Reduced from 0.12m
        self.second_stroke_amplitude = 0.105  # Reduced from 0.18m, maintains 1.5x ratio
        self.lift_height = 0.08  # Increased from 0.06m for better clearance
        
        # Base velocity parameters
        self.first_stroke_vx = 0.35  # Slightly reduced for smoother motion
        self.first_stroke_vy = 0.35
        self.second_stroke_vx = 0.52  # Reduced proportionally
        self.second_stroke_vy = 0.52
        self.hop_height = 0.04  # Slightly increased vertical hop
        
        # Initial body orientation (90° yaw for crab stance)
        self.initial_yaw = np.pi / 2.0
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = euler_to_quat(0.0, 0.0, self.initial_yaw)
        
    def reset(self, root_pos, root_quat):
        """Override reset to maintain sideways orientation."""
        self.root_pos = root_pos.copy()
        # Force sideways orientation
        self.root_quat = euler_to_quat(0.0, 0.0, self.initial_yaw)
        self.t = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent velocities.
        Yaw rate always zero to maintain perpendicular body orientation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Phase 0.0-0.3: First scuttle stroke
        if phase < 0.3:
            local_phase = phase / 0.3
            # Smooth velocity profile with cosine easing
            smooth = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vx = self.first_stroke_vx * smooth
            vy = self.first_stroke_vy * smooth
            # Slight upward velocity to prevent base settling
            vz = 0.01
            
        # Phase 0.3-0.5: Rapid reset with small hop
        elif phase < 0.5:
            local_phase = (phase - 0.3) / 0.2
            # Decay horizontal velocities smoothly
            decay = 1.0 - local_phase
            vx = self.first_stroke_vx * decay * 0.4
            vy = self.first_stroke_vy * decay * 0.4
            # Small vertical oscillation for hop
            vz = self.hop_height * np.sin(np.pi * local_phase) * 8.0
            
        # Phase 0.5-0.8: Second amplified stroke
        elif phase < 0.8:
            local_phase = (phase - 0.5) / 0.3
            # Smooth velocity profile with cosine easing
            smooth = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            vx = self.second_stroke_vx * smooth
            vy = self.second_stroke_vy * smooth
            # Slight upward velocity to prevent base settling
            vz = 0.015
            
        # Phase 0.8-1.0: Glide and stabilize
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth exponential decay to zero
            decay = np.exp(-5.0 * local_phase)
            vx = self.second_stroke_vx * decay * 0.3
            vy = self.second_stroke_vy * decay * 0.3
            # Settle back down
            vz = -0.01 * local_phase
        
        # Angular velocities always zero to maintain constant yaw
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        Compute foot position for each leg based on phase.
        Front legs (FL, FR) sweep backward during strokes.
        Rear legs (RL, RR) sweep forward during strokes.
        All legs lift together during reset phase.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.3: First scuttle stroke
        if phase < 0.3:
            local_phase = phase / 0.3
            # Smooth sweep with cosine easing to reduce joint stress
            smooth_phase = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            if is_front:
                # Front legs sweep backward (negative x)
                foot[0] -= self.first_stroke_amplitude * smooth_phase
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += self.first_stroke_amplitude * smooth_phase
                
        # Phase 0.3-0.5: Rapid reset with lift
        elif phase < 0.5:
            local_phase = (phase - 0.3) / 0.2
            # All legs lift and return to start
            if is_front:
                # Return forward - ensure complete return to base position
                sweep_offset = -self.first_stroke_amplitude
                return_phase = 0.5 * (1.0 - np.cos(np.pi * local_phase))
                foot[0] += sweep_offset * (1.0 - return_phase)
            else:
                # Return backward - ensure complete return to base position
                sweep_offset = self.first_stroke_amplitude
                return_phase = 0.5 * (1.0 - np.cos(np.pi * local_phase))
                foot[0] += sweep_offset * (1.0 - return_phase)
            # Parabolic lift with increased clearance
            foot[2] += self.lift_height * np.sin(np.pi * local_phase)
            
        # Phase 0.5-0.8: Second amplified stroke
        elif phase < 0.8:
            local_phase = (phase - 0.5) / 0.3
            # Smooth sweep with cosine easing
            smooth_phase = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            if is_front:
                # Front legs sweep backward with greater amplitude
                foot[0] -= self.second_stroke_amplitude * smooth_phase
            else:
                # Rear legs sweep forward with greater amplitude
                foot[0] += self.second_stroke_amplitude * smooth_phase
                
        # Phase 0.8-1.0: Glide and stabilize
        else:
            local_phase = (phase - 0.8) / 0.2
            # Return to neutral position smoothly with cosine easing
            return_phase = 0.5 * (1.0 - np.cos(np.pi * local_phase))
            if is_front:
                sweep_offset = -self.second_stroke_amplitude
                foot[0] += sweep_offset * return_phase
            else:
                sweep_offset = self.second_stroke_amplitude
                foot[0] += sweep_offset * return_phase
        
        return foot