from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet Bounce Diagonal: Dynamic zigzag locomotion with alternating diagonal bounces.
    
    The robot executes explosive diagonal bounces that ricochet left and right:
    - Phase [0.0-0.2]: Left compression with leftward yaw
    - Phase [0.2-0.4]: Right diagonal launch (airborne)
    - Phase [0.4-0.6]: Right compression with rightward yaw
    - Phase [0.6-0.8]: Left diagonal launch (airborne)
    - Phase [0.8-1.0]: Landing and yaw neutralization
    
    Contact pattern alternates between diagonal pairs (FL+RL) and (FR+RR).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced to prevent joint limit violations
        self.compression_depth = 0.03  # meters, reduced from 0.08
        self.lateral_amplitude = 0.06  # meters, lateral deviation in body frame
        self.forward_step = 0.08       # meters, forward progression per bounce
        
        # Velocity command magnitudes - reduced for smoother motion
        self.vx_forward = 0.7          # m/s, forward velocity during motion
        self.vx_launch = 1.0           # m/s, forward velocity during launch
        self.vy_lateral = 0.4          # m/s, lateral velocity magnitude
        self.vz_compression = -0.5     # m/s, downward velocity during compression
        self.vz_launch = 0.8           # m/s, upward velocity during launch
        self.yaw_rate_magnitude = 1.8  # rad/s, yaw rotation rate
        
        # Leg retraction during flight - small inward motion instead of extension
        self.flight_retraction = 0.02  # meters, slight inward retraction during flight
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocities to create ricochet bounce pattern with diagonal launches.
        Base motion is the primary mechanism for creating vertical excursion.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0-0.2]: Left compression with leftward yaw
        if 0.0 <= phase < 0.2:
            local_phase = phase / 0.2
            smooth = np.sin(np.pi * local_phase * 0.5)
            vx = self.vx_forward * (1.0 - 0.2 * smooth)
            vy = -self.vy_lateral * 0.25 * smooth
            vz = self.vz_compression * smooth
            yaw_rate = self.yaw_rate_magnitude * (1.0 - smooth)
        
        # Phase [0.2-0.4]: Right diagonal launch (airborne)
        elif 0.2 <= phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            smooth_up = np.cos(np.pi * local_phase * 0.5)
            vx = self.vx_launch
            vy = self.vy_lateral * (0.3 + 0.4 * local_phase)
            vz = self.vz_launch * smooth_up
            yaw_rate = -self.yaw_rate_magnitude * (0.5 * local_phase)
        
        # Phase [0.4-0.6]: Right compression with rightward yaw
        elif 0.4 <= phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            smooth = np.sin(np.pi * local_phase * 0.5)
            vx = self.vx_forward * (1.0 - 0.2 * smooth)
            vy = self.vy_lateral * 0.4 * (1.0 - local_phase)
            vz = self.vz_compression * smooth
            yaw_rate = -self.yaw_rate_magnitude * (1.0 - smooth)
        
        # Phase [0.6-0.8]: Left diagonal launch (airborne)
        elif 0.6 <= phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            smooth_up = np.cos(np.pi * local_phase * 0.5)
            vx = self.vx_launch
            vy = -self.vy_lateral * (0.3 + 0.4 * local_phase)
            vz = self.vz_launch * smooth_up
            yaw_rate = self.yaw_rate_magnitude * (0.5 * local_phase)
        
        # Phase [0.8-1.0]: Landing and yaw neutralization
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth = np.sin(np.pi * local_phase * 0.5)
            vx = self.vx_forward * (0.8 + 0.2 * local_phase)
            vy = -self.vy_lateral * 0.4 * (1.0 - local_phase)
            vz = self.vz_compression * smooth * 0.7
            yaw_rate = self.yaw_rate_magnitude * (1.0 - local_phase) * 0.4
        
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame.
        
        Key principle: Base motion handles vertical excursion. Foot z-positions in body frame
        remain nearly constant. Flight clearance comes from base rising, not feet retracting upward.
        Slight compression during stance and slight inward retraction during flight prevent joint limits.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        is_left_diagonal = is_left
        
        # Phase [0.0-0.2]: Left compression
        if 0.0 <= phase < 0.2:
            local_phase = phase / 0.2
            smooth = np.sin(np.pi * local_phase)
            
            if is_left_diagonal:
                # Stance: minimal compression, foot planted
                foot[2] -= self.compression_depth * smooth
                foot[0] -= self.forward_step * 0.15 * local_phase
                foot[1] += self.lateral_amplitude * 0.15 * smooth if is_left else 0
            else:
                # Swing: slight lift preparation
                foot[2] += self.compression_depth * 0.3 * local_phase
                foot[0] += self.forward_step * 0.2 * local_phase
        
        # Phase [0.2-0.4]: Right diagonal launch (all airborne)
        elif 0.2 <= phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Slight inward retraction during flight to prevent hyperextension
            # Base is rising via vz_launch, so feet should NOT move upward in body frame
            foot[2] += self.flight_retraction * np.sin(np.pi * local_phase)
            foot[0] += self.forward_step * 0.4 * (1.0 - local_phase)
            foot[1] += self.lateral_amplitude * (0.4 if is_left else -0.4) * np.sin(np.pi * local_phase * 0.5)
        
        # Phase [0.4-0.6]: Right compression
        elif 0.4 <= phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            smooth = np.sin(np.pi * local_phase)
            
            if not is_left_diagonal:
                # Stance: minimal compression, foot planted
                foot[2] -= self.compression_depth * smooth
                foot[0] -= self.forward_step * 0.15 * local_phase
                foot[1] -= self.lateral_amplitude * 0.15 * smooth if not is_left else 0
            else:
                # Swing: retract and prepare for next launch
                foot[2] += self.compression_depth * 0.3 * (1.0 - local_phase)
                foot[0] += self.forward_step * 0.2 * local_phase
        
        # Phase [0.6-0.8]: Left diagonal launch (all airborne)
        elif 0.6 <= phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Slight inward retraction during flight
            foot[2] += self.flight_retraction * np.sin(np.pi * local_phase)
            foot[0] += self.forward_step * 0.4 * (1.0 - local_phase)
            foot[1] += self.lateral_amplitude * (-0.4 if is_left else 0.4) * np.sin(np.pi * local_phase * 0.5)
        
        # Phase [0.8-1.0]: Landing and reset
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth = np.sin(np.pi * local_phase)
            
            if is_left_diagonal:
                # Stance: landing compression
                foot[2] -= self.compression_depth * smooth * 0.8
                foot[0] -= self.forward_step * 0.1 * local_phase
                # Neutralize lateral position gradually
                foot[1] *= (1.0 - 0.3 * local_phase)
            else:
                # Swing: return to neutral
                foot[2] += self.flight_retraction * 0.5 * (1.0 - local_phase)
                foot[0] += self.forward_step * 0.15 * (1.0 - local_phase)
                foot[1] *= (1.0 - 0.3 * local_phase)
        
        return foot