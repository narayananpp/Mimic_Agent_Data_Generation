from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet bounce diagonal skill with alternating left and right diagonal bounces.
    
    Motion pattern:
    - Phase 0.0-0.2: Left compression + left yaw (FL, RL stance)
    - Phase 0.2-0.4: Right diagonal launch (aerial)
    - Phase 0.4-0.6: Right compression + right yaw (FR, RR stance)
    - Phase 0.6-0.8: Left diagonal launch (aerial)
    - Phase 0.8-1.0: Landing prep + yaw neutralization (FL, RL landing)
    
    Reduced magnitudes to prevent ground penetration and joint limit violations.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Reduced base motion parameters to prevent excessive compression
        self.forward_velocity = 0.8
        self.lateral_velocity_amp = 0.3  # Reduced from 0.4
        self.vertical_velocity_launch = 0.9  # Reduced from 1.2
        self.vertical_velocity_compress = -0.35  # Reduced magnitude from -0.6
        
        # Reduced yaw parameters to lower hip angle demands
        self.yaw_rate_compress = 1.6  # Reduced from 2.1 to achieve ~20-25° instead of 30°
        
        # Reduced leg motion parameters to stay within joint limits
        self.compression_depth = 0.06  # Reduced from 0.12
        self.retraction_height = 0.09  # Reduced from 0.15
        self.extension_distance = 0.06  # Reduced from 0.08
        self.stance_clearance = 0.06  # Positive clearance baseline to prevent ground penetration
        
        # Smoothing parameter for phase transitions
        self.transition_blend = 0.08

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase with smoothed transitions.
        """
        
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Left compression + left yaw
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth compression velocity ramp
            compression_envelope = np.sin(local_phase * np.pi)
            vz = self.vertical_velocity_compress * compression_envelope
            yaw_rate = self.yaw_rate_compress * compression_envelope
            
        # Phase 0.2-0.4: Right diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth launch trajectory
            launch_envelope = np.cos(local_phase * np.pi * 0.5)
            vz = self.vertical_velocity_launch * launch_envelope
            vy = self.lateral_velocity_amp * np.sin(local_phase * np.pi)
            yaw_rate = 0.0
            
        # Phase 0.4-0.6: Right compression + right yaw
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            compression_envelope = np.sin(local_phase * np.pi)
            vz = self.vertical_velocity_compress * compression_envelope
            yaw_rate = -self.yaw_rate_compress * compression_envelope
            
        # Phase 0.6-0.8: Left diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            launch_envelope = np.cos(local_phase * np.pi * 0.5)
            vz = self.vertical_velocity_launch * launch_envelope
            vy = -self.lateral_velocity_amp * np.sin(local_phase * np.pi)
            yaw_rate = 0.0
            
        # Phase 0.8-1.0: Landing prep + yaw neutralization
        else:
            local_phase = (phase - 0.8) / 0.2
            descent_envelope = np.sin(local_phase * np.pi * 0.5)
            vz = self.vertical_velocity_compress * descent_envelope
            yaw_rate = self.yaw_rate_compress * descent_envelope
        
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def smooth_transition(self, phase, phase_start, phase_end):
        """Helper to create smooth transitions at phase boundaries."""
        if phase < phase_start:
            return 0.0
        elif phase > phase_end:
            return 1.0
        else:
            local = (phase - phase_start) / (phase_end - phase_start)
            return 0.5 * (1.0 - np.cos(local * np.pi))

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with positive clearance baseline
        and reduced motion magnitudes to prevent ground penetration and joint violations.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front_left = leg_name.startswith('FL')
        is_front_right = leg_name.startswith('FR')
        is_rear_left = leg_name.startswith('RL')
        is_rear_right = leg_name.startswith('RR')
        
        is_left_leg = is_front_left or is_rear_left
        is_right_leg = is_front_right or is_rear_right
        
        # Phase 0.0-0.2: Left compression (FL, RL stance)
        if phase < 0.2:
            local_phase = phase / 0.2
            compression_progress = np.sin(local_phase * np.pi * 0.5)
            
            if is_left_leg:
                # Stance: establish clearance, then compress smoothly
                foot[2] = foot[2] + self.stance_clearance - self.compression_depth * compression_progress
            else:
                # Swing: gentle retract with smooth transition from previous phase
                blend_in = self.smooth_transition(phase, 0.0, self.transition_blend)
                foot[2] += self.retraction_height * 0.4 * blend_in
                foot[0] -= self.extension_distance * 0.3 * blend_in
                
        # Phase 0.2-0.4: Right diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            aerial_progress = 0.5 * (1.0 - np.cos(local_phase * np.pi))
            
            if is_left_leg:
                # Extend away during flight, prepare for swing
                foot[0] += self.extension_distance * 0.4 * aerial_progress
                foot[1] -= self.extension_distance * 0.25 * aerial_progress
                foot[2] += self.retraction_height * 0.5 * aerial_progress
            else:
                # Right legs extend toward landing position
                foot[0] += self.extension_distance * 0.4 * aerial_progress
                foot[1] += self.extension_distance * 0.25 * aerial_progress
                foot[2] += self.retraction_height * 0.3 * (1.0 - aerial_progress)
                
        # Phase 0.4-0.6: Right compression (FR, RR stance)
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            compression_progress = np.sin(local_phase * np.pi * 0.5)
            
            if is_right_leg:
                # Stance: establish clearance, then compress
                foot[2] = foot[2] + self.stance_clearance - self.compression_depth * compression_progress
            else:
                # Swing: gentle retract
                blend_in = self.smooth_transition(phase, 0.4, 0.4 + self.transition_blend)
                foot[2] += self.retraction_height * 0.4 * blend_in
                foot[0] -= self.extension_distance * 0.3 * blend_in
                
        # Phase 0.6-0.8: Left diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            aerial_progress = 0.5 * (1.0 - np.cos(local_phase * np.pi))
            
            if is_right_leg:
                # Extend away during flight
                foot[0] += self.extension_distance * 0.4 * aerial_progress
                foot[1] += self.extension_distance * 0.25 * aerial_progress
                foot[2] += self.retraction_height * 0.5 * aerial_progress
            else:
                # Left legs extend toward landing position
                foot[0] += self.extension_distance * 0.4 * aerial_progress
                foot[1] -= self.extension_distance * 0.25 * aerial_progress
                foot[2] += self.retraction_height * 0.3 * (1.0 - aerial_progress)
                
        # Phase 0.8-1.0: Landing prep (FL, RL landing)
        else:
            local_phase = (phase - 0.8) / 0.2
            landing_progress = np.sin(local_phase * np.pi * 0.5)
            
            if is_left_leg:
                # Smooth landing with progressive compression
                foot[2] = foot[2] + self.stance_clearance * (1.0 - landing_progress * 0.5)
                foot[0] -= self.extension_distance * 0.2 * landing_progress
            else:
                # Right legs retract for next cycle
                foot[2] += self.retraction_height * 0.3 * landing_progress
                foot[0] -= self.extension_distance * 0.2 * landing_progress
        
        return foot