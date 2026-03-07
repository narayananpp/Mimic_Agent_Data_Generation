from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Diagonal ricochet bounce locomotion with alternating yaw-biased explosive launches.
    
    Motion pattern:
    - Phase 0.0-0.2: Left compression with left yaw rotation
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
        
        # Motion parameters - reduced to prevent joint limit violations
        self.compression_depth = 0.07  # Reduced from 0.12 to limit leg compression
        self.launch_height = 0.25      # Maximum aerial height
        self.step_height = 0.10        # Reduced from 0.15 to limit retraction range
        
        # Velocity parameters - rebalanced for net-zero vertical displacement
        self.forward_velocity = 0.8    # Sustained forward speed
        self.lateral_velocity = 0.5    # Lateral component for diagonal launches
        self.launch_velocity = 2.2     # Increased from 1.5 to prevent base sinking
        self.compression_velocity = -0.55  # Reduced from -0.8 to limit downward integration
        
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
        
        Rebalanced vertical velocity profile to maintain stable height without height envelope.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        roll_rate = 0.0
        
        # Phase 0.0-0.2: Left compression with left yaw
        if phase < 0.2:
            sub_phase = phase / 0.2
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)
            vy = -0.1 * sub_phase
            # Smooth compression with reduced magnitude
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)
            yaw_rate = -self.yaw_rate_magnitude
            roll_rate = -0.3 * sub_phase
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial - extended upward phase
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)
            vy = self.lateral_velocity * sub_phase
            
            # Extended upward velocity duration to 50% of aerial phase
            if sub_phase < 0.5:
                # Smooth launch with cosine decay
                vz = self.launch_velocity * np.cos(np.pi * sub_phase / 1.0)
            else:
                # Minimal downward velocity during ballistic phase
                vz = -0.15 * (sub_phase - 0.5) / 0.5
            
            yaw_rate = self.yaw_rate_magnitude * 0.7
            roll_rate = 0.0
        
        # Phase 0.4-0.6: Right compression with right yaw
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)
            vy = 0.1 * sub_phase
            # Smooth compression with reduced magnitude
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)
            yaw_rate = self.yaw_rate_magnitude
            roll_rate = 0.3 * sub_phase
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial - extended upward phase
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)
            vy = -self.lateral_velocity * sub_phase
            
            # Extended upward velocity duration to 50% of aerial phase
            if sub_phase < 0.5:
                # Smooth launch with cosine decay
                vz = self.launch_velocity * np.cos(np.pi * sub_phase / 1.0)
            else:
                # Minimal downward velocity during ballistic phase
                vz = -0.15 * (sub_phase - 0.5) / 0.5
            
            yaw_rate = -self.yaw_rate_magnitude * 0.7
            roll_rate = 0.0
        
        # Phase 0.8-1.0: Landing preparation - stabilize height instead of compressing
        else:
            sub_phase = (phase - 0.8) / 0.2
            vx = self.forward_velocity * (0.9 + 0.1 * sub_phase)
            vy = -0.1 * (1.0 - sub_phase)
            # Small upward velocity to counteract any residual sinking
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
        
        Smoothed transitions to reduce joint velocity spikes and prevent limit violations.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 0.0-0.2: Left compression (left legs dominant)
        if phase < 0.2:
            sub_phase = phase / 0.2
            compression_factor = 1.0 if is_left_leg else 0.7
            # Smooth sinusoidal compression
            foot[2] += self.compression_depth * compression_factor * np.sin(np.pi * sub_phase)
            foot[0] -= 0.02 * sub_phase  # Reduced backward shift
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            
            # Extended smooth extension transition (35% instead of 25%)
            if sub_phase < 0.35:
                # Gradual extension using cosine smoothing
                extension = 0.5 * (1.0 - np.cos(np.pi * sub_phase / 0.35))
                foot[2] += self.compression_depth * (1.0 - extension) * (1.0 if is_left_leg else 0.7)
            else:
                # Smooth aerial retraction
                retract = (sub_phase - 0.35) / 0.65
                foot[2] -= self.step_height * np.sin(np.pi * retract * 0.8)  # Reduced peak
                foot[0] += 0.06 * retract  # Reduced forward swing
                
                # Right legs extend slightly more for landing
                if not is_left_leg:
                    foot[0] += 0.02 * retract
        
        # Phase 0.4-0.6: Right compression (right legs dominant)
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            compression_factor = 1.0 if not is_left_leg else 0.7
            # Smooth sinusoidal compression
            foot[2] += self.compression_depth * compression_factor * np.sin(np.pi * sub_phase)
            foot[0] -= 0.02 * sub_phase  # Reduced backward shift
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            
            # Extended smooth extension transition (35% instead of 25%)
            if sub_phase < 0.35:
                # Gradual extension using cosine smoothing
                extension = 0.5 * (1.0 - np.cos(np.pi * sub_phase / 0.35))
                foot[2] += self.compression_depth * (1.0 - extension) * (1.0 if not is_left_leg else 0.7)
            else:
                # Smooth aerial retraction
                retract = (sub_phase - 0.35) / 0.65
                foot[2] -= self.step_height * np.sin(np.pi * retract * 0.8)  # Reduced peak
                foot[0] += 0.06 * retract  # Reduced forward swing
                
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