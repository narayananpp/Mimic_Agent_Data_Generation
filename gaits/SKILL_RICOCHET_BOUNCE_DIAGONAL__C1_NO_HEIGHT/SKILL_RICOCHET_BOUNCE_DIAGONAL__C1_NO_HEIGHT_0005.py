from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal ricochet bounce locomotion with alternating yaw-biased explosive launches.
    
    Motion pattern:
    - Phase 0.0-0.1: Initialization - lift base to safe height
    - Phase 0.1-0.2: Left compression with left yaw rotation
    - Phase 0.2-0.4: Right-diagonal launch and aerial phase
    - Phase 0.4-0.6: Right compression with right yaw rotation
    - Phase 0.6-0.8: Left-diagonal launch and aerial phase
    - Phase 0.8-1.0: Landing preparation and yaw neutralization
    
    Creates zigzag forward trajectory through alternating diagonal bounces.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # ~1.25 seconds per full zigzag cycle
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - reduced and reshaped to prevent knee violations
        self.compression_depth = 0.03  # Reduced vertical component
        self.launch_height = 0.25      # Maximum aerial height
        self.step_height = 0.05        # Reduced aerial retraction
        
        # Velocity parameters - rebalanced for net-zero vertical displacement
        self.forward_velocity = 0.8    # Sustained forward speed
        self.lateral_velocity = 0.5    # Lateral component for diagonal launches
        self.launch_velocity = 2.2     # Upward launch velocity
        self.compression_velocity = -0.40  # Controlled base sinking
        
        # Yaw parameters
        self.yaw_amplitude = 0.52      # ~30 degrees in radians
        self.yaw_rate_magnitude = 5.0  # Yaw angular velocity magnitude
        
        # Time tracking
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        
        Includes extended initialization period and modulated yaw during compression.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        roll_rate = 0.0
        
        # Phase 0.0-0.1: Pure initialization - lift base before any compression
        if phase < 0.1:
            sub_phase = phase / 0.1
            vx = self.forward_velocity * 0.5
            vy = 0.0
            vz = 0.8 * (1.0 - sub_phase)  # Strong upward lift
            yaw_rate = 0.0
            roll_rate = 0.0
        
        # Phase 0.1-0.2: Left compression with left yaw (modulated)
        elif phase < 0.2:
            sub_phase = (phase - 0.1) / 0.1
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)
            vy = -0.1 * sub_phase
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)
            # Reduced yaw rate during peak compression
            yaw_rate = -self.yaw_rate_magnitude * (1.0 - 0.3 * np.sin(np.pi * sub_phase))
            roll_rate = -0.3 * sub_phase
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)
            vy = self.lateral_velocity * sub_phase
            
            if sub_phase < 0.5:
                vz = self.launch_velocity * np.cos(np.pi * sub_phase / 1.0)
            else:
                vz = -0.15 * (sub_phase - 0.5) / 0.5
            
            yaw_rate = self.yaw_rate_magnitude * 0.7
            roll_rate = 0.0
        
        # Phase 0.4-0.6: Right compression with right yaw (modulated)
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)
            vy = 0.1 * sub_phase
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)
            # Reduced yaw rate during peak compression
            yaw_rate = self.yaw_rate_magnitude * (1.0 - 0.3 * np.sin(np.pi * sub_phase))
            roll_rate = 0.3 * sub_phase
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)
            vy = -self.lateral_velocity * sub_phase
            
            if sub_phase < 0.5:
                vz = self.launch_velocity * np.cos(np.pi * sub_phase / 1.0)
            else:
                vz = -0.15 * (sub_phase - 0.5) / 0.5
            
            yaw_rate = -self.yaw_rate_magnitude * 0.7
            roll_rate = 0.0
        
        # Phase 0.8-1.0: Landing preparation
        else:
            sub_phase = (phase - 0.8) / 0.2
            vx = self.forward_velocity * (0.9 + 0.1 * sub_phase)
            vy = -0.1 * (1.0 - sub_phase)
            vz = 0.2 * (1.0 - sub_phase)
            yaw_rate = -self.yaw_rate_magnitude * 0.5
            roll_rate = 0.0
        
        # Apply velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, 0.0, yaw_rate])
        
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
        Compute foot position in BODY frame with multi-axis trajectory reshaping.
        
        Uses combined vertical, horizontal, and lateral motion to distribute kinematic
        demand across multiple joints and prevent knee limit violations.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name in ['FL', 'RL']
        is_front_leg = leg_name in ['FL', 'FR']
        
        # Phase 0.0-0.1: Initialization - maintain neutral stance
        if phase < 0.1:
            pass
        
        # Phase 0.1-0.2: Left compression with multi-axis trajectory
        elif phase < 0.2:
            sub_phase = (phase - 0.1) / 0.1
            compression_factor = 1.0 if is_left_leg else 0.8
            
            sin_factor = np.sin(np.pi * sub_phase)
            
            # Reduced vertical compression
            foot[2] += self.compression_depth * compression_factor * sin_factor
            
            # Increased backward motion to distribute leg shortening
            foot[0] -= 0.04 * sin_factor
            
            # Lateral outward motion to reduce knee flexion demand
            lateral_scale = 1.0 + 0.08 * sin_factor
            foot[1] *= lateral_scale
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial with circular retraction
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            
            # Extended extension phase (0.0-0.50)
            if sub_phase < 0.50:
                extension = 0.5 * (1.0 - np.cos(np.pi * sub_phase / 0.50))
                foot[2] += self.compression_depth * (1.0 - extension) * (1.0 if is_left_leg else 0.8)
                # Unwind backward motion
                foot[0] -= 0.04 * (1.0 - extension) * np.sin(np.pi * (1.0 - extension))
            
            # Neutral hold (0.50-0.55)
            elif sub_phase < 0.55:
                pass
            
            # Aerial retraction with inward lateral motion (0.55-1.0)
            else:
                retract = (sub_phase - 0.55) / 0.45
                retract_factor = np.sin(np.pi * retract * 0.6)
                
                # Reduced vertical retraction
                foot[2] -= self.step_height * retract_factor
                
                # Inward lateral motion to reduce knee flexion
                lateral_scale = 1.0 - 0.10 * retract_factor
                foot[1] *= lateral_scale
                
                # Reduced forward swing
                foot[0] += 0.03 * retract
                
                if not is_left_leg:
                    foot[0] += 0.015 * retract
        
        # Phase 0.4-0.6: Right compression with multi-axis trajectory
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            compression_factor = 1.0 if not is_left_leg else 0.8
            
            sin_factor = np.sin(np.pi * sub_phase)
            
            # Reduced vertical compression
            foot[2] += self.compression_depth * compression_factor * sin_factor
            
            # Increased backward motion
            foot[0] -= 0.04 * sin_factor
            
            # Lateral outward motion
            lateral_scale = 1.0 + 0.08 * sin_factor
            foot[1] *= lateral_scale
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial with circular retraction
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            
            # Extended extension phase (0.0-0.50)
            if sub_phase < 0.50:
                extension = 0.5 * (1.0 - np.cos(np.pi * sub_phase / 0.50))
                foot[2] += self.compression_depth * (1.0 - extension) * (1.0 if not is_left_leg else 0.8)
                # Unwind backward motion
                foot[0] -= 0.04 * (1.0 - extension) * np.sin(np.pi * (1.0 - extension))
            
            # Neutral hold (0.50-0.55)
            elif sub_phase < 0.55:
                pass
            
            # Aerial retraction with inward lateral motion (0.55-1.0)
            else:
                retract = (sub_phase - 0.55) / 0.45
                retract_factor = np.sin(np.pi * retract * 0.6)
                
                # Reduced vertical retraction
                foot[2] -= self.step_height * retract_factor
                
                # Inward lateral motion
                lateral_scale = 1.0 - 0.10 * retract_factor
                foot[1] *= lateral_scale
                
                # Reduced forward swing
                foot[0] += 0.03 * retract
                
                if is_left_leg:
                    foot[0] += 0.015 * retract
        
        # Phase 0.8-1.0: Landing preparation with smooth extension
        else:
            sub_phase = (phase - 0.8) / 0.2
            
            extension_factor = 0.5 * (1.0 - np.cos(np.pi * sub_phase))
            
            # Smooth transition from retracted to extended
            foot[2] -= self.step_height * 0.3 * (1.0 - extension_factor)
            
            # Gentle compression as contact establishes
            foot[2] += self.compression_depth * 0.2 * extension_factor
            
            # Position forward for next cycle
            foot[0] += 0.01 * sub_phase
        
        return foot