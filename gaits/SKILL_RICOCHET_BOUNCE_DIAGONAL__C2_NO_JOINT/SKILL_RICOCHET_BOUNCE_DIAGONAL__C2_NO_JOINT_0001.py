from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet Bounce Diagonal gait: alternating diagonal bounces creating zigzag locomotion.
    
    Motion structure:
    - Phase [0.0-0.2]: Left compression with left yaw (FL+RL stance)
    - Phase [0.2-0.4]: Right launch and flight (all legs airborne)
    - Phase [0.4-0.6]: Right compression with right yaw (FR+RR stance)
    - Phase [0.6-0.8]: Left launch and flight (all legs airborne)
    - Phase [0.8-1.0]: Landing preparation (FL+RL stance)
    
    Base motion: alternating diagonal launches with yaw modulation
    Leg motion: diagonal pair coordination with aerial phases
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency (Hz)
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_height = 0.12  # Vertical compression during loading
        self.launch_height = 0.15  # Leg extension during flight
        self.lateral_retraction = 0.08  # Lateral leg retraction during swing
        
        # Base velocity parameters
        self.vx_compression = 0.3  # Forward velocity during compression
        self.vx_launch = 1.5  # Forward velocity during launch
        self.vy_diagonal = 0.6  # Lateral velocity for diagonal trajectory
        self.vz_compression = -0.8  # Downward velocity during compression
        self.vz_launch = 1.2  # Upward velocity during launch
        
        # Yaw parameters
        self.yaw_rate_compression = 2.5  # Yaw rate during compression (rad/s)
        self.yaw_target = np.deg2rad(30)  # Target yaw angle accumulation
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase tracking for smooth transitions
        self.current_yaw = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base motion with phase-dependent velocities and yaw rates.
        Creates alternating diagonal bounces with yaw modulation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase [0.0-0.2]: Left compression with left yaw
        if 0.0 <= phase < 0.2:
            local_phase = phase / 0.2
            vx = self.vx_compression * (1.0 - 0.5 * local_phase)
            vy = 0.0
            vz = self.vz_compression * np.sin(np.pi * local_phase)
            yaw_rate = -self.yaw_rate_compression
        
        # Phase [0.2-0.4]: Right launch and flight
        elif 0.2 <= phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            vx = self.vx_compression + (self.vx_launch - self.vx_compression) * local_phase
            vy = self.vy_diagonal * np.sin(np.pi * local_phase)
            vz = self.vz_launch * np.sin(np.pi * local_phase)
            # Neutralize yaw rate during flight
            yaw_rate = -self.yaw_rate_compression * (1.0 - local_phase)
        
        # Phase [0.4-0.6]: Right compression with right yaw
        elif 0.4 <= phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            vx = self.vx_launch * (1.0 - 0.6 * local_phase)
            vy = self.vy_diagonal * (1.0 - local_phase)
            vz = self.vz_compression * np.sin(np.pi * local_phase)
            yaw_rate = self.yaw_rate_compression
        
        # Phase [0.6-0.8]: Left launch and flight
        elif 0.6 <= phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vx = self.vx_compression + (self.vx_launch - self.vx_compression) * local_phase
            vy = -self.vy_diagonal * np.sin(np.pi * local_phase)
            vz = self.vz_launch * np.sin(np.pi * local_phase)
            # Neutralize yaw rate during flight
            yaw_rate = self.yaw_rate_compression * (1.0 - local_phase)
        
        # Phase [0.8-1.0]: Landing preparation and neutralization
        else:
            local_phase = (phase - 0.8) / 0.2
            vx = self.vx_launch * (1.0 - 0.6 * local_phase)
            vy = -self.vy_diagonal * (1.0 - local_phase)
            vz = self.vz_compression * np.sin(np.pi * local_phase) * (1.0 - local_phase)
            yaw_rate = -self.yaw_rate_compression * local_phase
        
        # Set velocities
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
        Compute foot position in body frame for each leg based on phase.
        
        Diagonal pair coordination:
        - FL+RL: stance during [0.0-0.2] and [0.8-1.0], swing otherwise
        - FR+RR: stance during [0.4-0.6], swing otherwise
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front_left = leg_name.startswith('FL')
        is_front_right = leg_name.startswith('FR')
        is_rear_left = leg_name.startswith('RL')
        is_rear_right = leg_name.startswith('RR')
        
        # Left diagonal group (FL, RL)
        if is_front_left or is_rear_left:
            # Phase [0.0-0.2]: Stance - left compression
            if 0.0 <= phase < 0.2:
                local_phase = phase / 0.2
                foot[2] -= self.compression_height * np.sin(np.pi * local_phase)
            
            # Phase [0.2-0.4]: Swing - extend during right launch
            elif 0.2 <= phase < 0.4:
                local_phase = (phase - 0.2) / 0.2
                foot[2] += self.launch_height * np.sin(np.pi * local_phase)
                if is_front_left:
                    foot[0] += 0.05 * local_phase
                else:
                    foot[0] -= 0.05 * local_phase
                foot[1] += self.lateral_retraction * np.sin(np.pi * local_phase)
            
            # Phase [0.4-0.6]: Swing - retract during right compression
            elif 0.4 <= phase < 0.6:
                local_phase = (phase - 0.4) / 0.2
                foot[2] += self.launch_height * (1.0 - local_phase)
                if is_front_left:
                    foot[0] += 0.05 * (1.0 - local_phase)
                else:
                    foot[0] -= 0.05 * (1.0 - local_phase)
                foot[1] += self.lateral_retraction * (1.0 - local_phase)
            
            # Phase [0.6-0.8]: Swing - extend during left launch
            elif 0.6 <= phase < 0.8:
                local_phase = (phase - 0.6) / 0.2
                foot[2] += self.launch_height * np.sin(np.pi * local_phase)
                if is_front_left:
                    foot[0] += 0.05 * local_phase
                else:
                    foot[0] -= 0.05 * local_phase
                foot[1] += self.lateral_retraction * np.sin(np.pi * local_phase)
            
            # Phase [0.8-1.0]: Stance - landing and preparation
            else:
                local_phase = (phase - 0.8) / 0.2
                foot[2] += self.launch_height * (1.0 - local_phase)
                foot[2] -= self.compression_height * local_phase * np.sin(np.pi * local_phase)
                if is_front_left:
                    foot[0] += 0.05 * (1.0 - local_phase)
                else:
                    foot[0] -= 0.05 * (1.0 - local_phase)
                foot[1] += self.lateral_retraction * (1.0 - local_phase)
        
        # Right diagonal group (FR, RR)
        elif is_front_right or is_rear_right:
            # Phase [0.0-0.2]: Swing - retract during left compression
            if 0.0 <= phase < 0.2:
                local_phase = phase / 0.2
                foot[1] -= self.lateral_retraction * local_phase
            
            # Phase [0.2-0.4]: Swing - extend during right launch
            elif 0.2 <= phase < 0.4:
                local_phase = (phase - 0.2) / 0.2
                foot[2] += self.launch_height * np.sin(np.pi * local_phase)
                if is_front_right:
                    foot[0] += 0.05 * local_phase
                else:
                    foot[0] -= 0.05 * local_phase
                foot[1] -= self.lateral_retraction * (1.0 - np.sin(np.pi * local_phase))
            
            # Phase [0.4-0.6]: Stance - right compression
            elif 0.4 <= phase < 0.6:
                local_phase = (phase - 0.4) / 0.2
                foot[2] += self.launch_height * (1.0 - local_phase)
                foot[2] -= self.compression_height * local_phase * np.sin(np.pi * local_phase)
                if is_front_right:
                    foot[0] += 0.05 * (1.0 - local_phase)
                else:
                    foot[0] -= 0.05 * (1.0 - local_phase)
            
            # Phase [0.6-0.8]: Swing - extend during left launch
            elif 0.6 <= phase < 0.8:
                local_phase = (phase - 0.6) / 0.2
                foot[2] += self.launch_height * np.sin(np.pi * local_phase)
                if is_front_right:
                    foot[0] += 0.05 * local_phase
                else:
                    foot[0] -= 0.05 * local_phase
                foot[1] -= self.lateral_retraction * np.sin(np.pi * local_phase)
            
            # Phase [0.8-1.0]: Swing - retract for next cycle
            else:
                local_phase = (phase - 0.8) / 0.2
                foot[2] += self.launch_height * (1.0 - local_phase)
                if is_front_right:
                    foot[0] += 0.05 * (1.0 - local_phase)
                else:
                    foot[0] -= 0.05 * (1.0 - local_phase)
                foot[1] -= self.lateral_retraction * (1.0 - local_phase)
        
        return foot