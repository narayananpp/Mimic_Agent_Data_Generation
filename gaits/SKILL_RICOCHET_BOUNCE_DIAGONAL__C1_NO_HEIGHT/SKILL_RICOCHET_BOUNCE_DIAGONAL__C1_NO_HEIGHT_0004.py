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
        
        # Motion parameters - tuned to prevent joint violations and ground penetration
        self.compression_depth = 0.05  # Reduced from 0.07 to limit knee flexion demand
        self.launch_height = 0.25      # Maximum aerial height
        self.step_height = 0.07        # Aerial retraction height
        
        # Velocity parameters - rebalanced for net-zero vertical displacement
        self.forward_velocity = 0.8    # Sustained forward speed
        self.lateral_velocity = 0.5    # Lateral component for diagonal launches
        self.launch_velocity = 2.2     # Upward launch velocity
        self.compression_velocity = -0.40  # Reduced from -0.55 to limit base sinking
        
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
        
        Includes extended initialization period to establish safe height before compression.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        roll_rate = 0.0
        
        # Phase 0.0-0.1: Pure initialization - lift base before any compression
        if phase < 0.1:
            sub_phase = phase / 0.1
            vx = self.forward_velocity * 0.5  # Reduced forward speed during init
            vy = 0.0
            # Strong upward velocity to establish safe height
            vz = 0.8 * (1.0 - sub_phase)
            yaw_rate = 0.0
            roll_rate = 0.0
        
        # Phase 0.1-0.2: Left compression with left yaw
        elif phase < 0.2:
            sub_phase = (phase - 0.1) / 0.1
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)
            vy = -0.1 * sub_phase
            # Smooth compression with reduced magnitude
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)
            yaw_rate = -self.yaw_rate_magnitude
            roll_rate = -0.3 * sub_phase
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)
            vy = self.lateral_velocity * sub_phase
            
            # Extended upward velocity duration
            if sub_phase < 0.5:
                vz = self.launch_velocity * np.cos(np.pi * sub_phase / 1.0)
            else:
                vz = -0.15 * (sub_phase - 0.5) / 0.5
            
            yaw_rate = self.yaw_rate_magnitude * 0.7
            roll_rate = 0.0
        
        # Phase 0.4-0.6: Right compression with right yaw
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)
            vy = 0.1 * sub_phase
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)
            yaw_rate = self.yaw_rate_magnitude
            roll_rate = 0.3 * sub_phase
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)
            vy = -self.lateral_velocity * sub_phase
            
            # Extended upward velocity duration
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
        Compute foot position in BODY frame with asymmetric loading and aerial retraction.
        
        Delayed compression onset and reduced magnitudes to prevent joint violations.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 0.0-0.1: Initialization - maintain neutral stance
        if phase < 0.1:
            pass  # Keep feet at base positions during initialization lift
        
        # Phase 0.1-0.2: Left compression (left legs dominant)
        elif phase < 0.2:
            sub_phase = (phase - 0.1) / 0.1
            # Narrowed compression factor difference: 1.0 vs 0.8 instead of 1.0 vs 0.7
            compression_factor = 1.0 if is_left_leg else 0.8
            
            # Smooth sinusoidal compression
            foot[2] += self.compression_depth * compression_factor * np.sin(np.pi * sub_phase)
            foot[0] -= 0.02 * sub_phase
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            
            # Extended extension phase (0.0-0.40)
            if sub_phase < 0.40:
                # Gradual extension using cosine smoothing
                extension = 0.5 * (1.0 - np.cos(np.pi * sub_phase / 0.40))
                foot[2] += self.compression_depth * (1.0 - extension) * (1.0 if is_left_leg else 0.8)
            
            # Neutral hold (0.40-0.45)
            elif sub_phase < 0.45:
                pass  # Maintain extended position
            
            # Aerial retraction phase (0.45-1.0)
            else:
                retract = (sub_phase - 0.45) / 0.55
                # Reduced retraction magnitude
                foot[2] -= self.step_height * np.sin(np.pi * retract * 0.6)
                # Reduced forward swing
                foot[0] += 0.04 * retract
                
                # Right legs extend slightly more for landing
                if not is_left_leg:
                    foot[0] += 0.02 * retract
        
        # Phase 0.4-0.6: Right compression (right legs dominant)
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            compression_factor = 1.0 if not is_left_leg else 0.8
            
            # Smooth sinusoidal compression
            foot[2] += self.compression_depth * compression_factor * np.sin(np.pi * sub_phase)
            foot[0] -= 0.02 * sub_phase
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            
            # Extended extension phase (0.0-0.40)
            if sub_phase < 0.40:
                # Gradual extension using cosine smoothing
                extension = 0.5 * (1.0 - np.cos(np.pi * sub_phase / 0.40))
                foot[2] += self.compression_depth * (1.0 - extension) * (1.0 if not is_left_leg else 0.8)
            
            # Neutral hold (0.40-0.45)
            elif sub_phase < 0.45:
                pass  # Maintain extended position
            
            # Aerial retraction phase (0.45-1.0)
            else:
                retract = (sub_phase - 0.45) / 0.55
                # Reduced retraction magnitude
                foot[2] -= self.step_height * np.sin(np.pi * retract * 0.6)
                # Reduced forward swing
                foot[0] += 0.04 * retract
                
                # Left legs extend slightly more for landing
                if is_left_leg:
                    foot[0] += 0.02 * retract
        
        # Phase 0.8-1.0: Landing preparation
        else:
            sub_phase = (phase - 0.8) / 0.2
            
            # Smooth extension for landing using cosine
            extension_factor = 0.5 * (1.0 - np.cos(np.pi * sub_phase))
            foot[2] -= self.step_height * 0.3 * (1.0 - extension_factor)
            
            # Gentle compression as contact establishes
            foot[2] += self.compression_depth * 0.25 * extension_factor
            
            # Position forward for next cycle
            foot[0] += 0.015 * sub_phase
        
        return foot