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
    
    Base motion combines forward velocity with alternating lateral velocity
    and oscillating yaw rotation (~±30°).
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Base motion parameters
        self.forward_velocity = 0.8  # Sustained forward velocity
        self.lateral_velocity_amp = 0.4  # Lateral velocity during launches
        self.vertical_velocity_launch = 1.2  # Upward velocity during launch
        self.vertical_velocity_compress = -0.6  # Downward velocity during compression
        
        # Yaw rotation parameters
        # Target ~30° rotation: 30° = π/6 ≈ 0.524 rad
        # During 0.2 phase duration at 0.8 Hz: Δt = 0.2/0.8 = 0.25s
        # yaw_rate = 0.524 / 0.25 = 2.1 rad/s
        self.yaw_rate_compress = 2.1  # Yaw rate during compression phases
        
        # Leg motion parameters
        self.compression_depth = 0.12  # Vertical compression during stance
        self.retraction_height = 0.15  # Leg retraction during swing
        self.extension_distance = 0.08  # Horizontal extension during aerial

    def update_base_motion(self, phase, dt):
        """
        Update base velocities based on phase.
        
        Phase breakdown:
        - 0.0-0.2: Compress down, yaw left
        - 0.2-0.4: Launch up+forward+right
        - 0.4-0.6: Compress down, yaw right
        - 0.6-0.8: Launch up+forward+left
        - 0.8-1.0: Descend, neutralize yaw
        """
        
        vx = self.forward_velocity
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 0.0-0.2: Left compression + left yaw
        if phase < 0.2:
            vz = self.vertical_velocity_compress
            yaw_rate = self.yaw_rate_compress  # Positive = left yaw
            
        # Phase 0.2-0.4: Right diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Peak velocity at start, decay toward end
            vz = self.vertical_velocity_launch * (1.0 - local_phase)
            vy = self.lateral_velocity_amp  # Rightward
            yaw_rate = 0.0  # Maintain accumulated yaw
            
        # Phase 0.4-0.6: Right compression + right yaw
        elif phase < 0.6:
            vz = self.vertical_velocity_compress
            yaw_rate = -self.yaw_rate_compress  # Negative = right yaw
            
        # Phase 0.6-0.8: Left diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vz = self.vertical_velocity_launch * (1.0 - local_phase)
            vy = -self.lateral_velocity_amp  # Leftward
            yaw_rate = 0.0  # Maintain accumulated yaw
            
        # Phase 0.8-1.0: Landing prep + yaw neutralization
        else:
            local_phase = (phase - 0.8) / 0.2
            vz = self.vertical_velocity_compress * local_phase  # Gradual descent
            # Neutralize yaw: reverse previous right yaw
            yaw_rate = self.yaw_rate_compress * local_phase
        
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
        Compute foot position in body frame based on phase and leg.
        
        Contact schedule:
        - FL, RL: stance during 0.0-0.2 and 0.8-1.0
        - FR, RR: stance during 0.4-0.6
        - All legs: aerial during 0.2-0.4 and 0.6-0.8
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
            if is_left_leg:
                # Stance: compress downward
                foot[2] -= self.compression_depth * local_phase
            else:
                # Swing: retract toward body
                foot[2] += self.retraction_height * (1.0 - local_phase)
                
        # Phase 0.2-0.4: Right diagonal launch (aerial)
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            if is_left_leg:
                # Extend forward and laterally during flight
                foot[0] += self.extension_distance * local_phase
                foot[1] -= self.extension_distance * 0.5 * local_phase  # Slight lateral
                foot[2] += self.retraction_height * (1.0 - local_phase * 0.5)
            else:
                # Right legs extend toward landing
                foot[0] += self.extension_distance * local_phase
                foot[1] += self.extension_distance * 0.5 * local_phase
                foot[2] += self.retraction_height * (1.0 - local_phase)
                
        # Phase 0.4-0.6: Right compression (FR, RR stance)
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            if is_right_leg:
                # Stance: compress downward
                foot[2] -= self.compression_depth * local_phase
            else:
                # Swing: retract toward body
                foot[2] += self.retraction_height * (1.0 - local_phase)
                
        # Phase 0.6-0.8: Left diagonal launch (aerial)
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            if is_right_leg:
                # Extend during flight
                foot[0] += self.extension_distance * local_phase
                foot[1] += self.extension_distance * 0.5 * local_phase
                foot[2] += self.retraction_height * (1.0 - local_phase * 0.5)
            else:
                # Left legs extend toward landing
                foot[0] += self.extension_distance * local_phase
                foot[1] -= self.extension_distance * 0.5 * local_phase
                foot[2] += self.retraction_height * (1.0 - local_phase)
                
        # Phase 0.8-1.0: Landing prep (FL, RL landing)
        else:
            local_phase = (phase - 0.8) / 0.2
            if is_left_leg:
                # Extend to landing position with compression
                foot[2] -= self.compression_depth * local_phase
            else:
                # Right legs retract
                foot[2] += self.retraction_height * local_phase
        
        return foot