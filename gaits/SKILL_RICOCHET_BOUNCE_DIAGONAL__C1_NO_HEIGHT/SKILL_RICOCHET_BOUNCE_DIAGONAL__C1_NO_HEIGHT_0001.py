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
        
        # Motion parameters
        self.compression_depth = 0.12  # Vertical compression during stance
        self.launch_height = 0.25      # Maximum aerial height
        self.step_height = 0.15        # Leg retraction height during aerial phase
        
        # Velocity parameters
        self.forward_velocity = 0.8    # Sustained forward speed
        self.lateral_velocity = 0.5    # Lateral component for diagonal launches
        self.launch_velocity = 1.5     # Vertical launch velocity
        self.compression_velocity = -0.8  # Downward velocity during compression
        
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
        
        Creates compression-launch-aerial-landing cycle with yaw alternation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        roll_rate = 0.0
        
        # Phase 0.0-0.2: Left compression with left yaw
        if phase < 0.2:
            sub_phase = phase / 0.2
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)  # Decelerate slightly
            vy = -0.1 * sub_phase  # Slight left drift
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)  # Smooth compression
            yaw_rate = -self.yaw_rate_magnitude  # Rotate left
            roll_rate = -0.3 * sub_phase  # Slight left roll bias
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)  # Accelerate forward
            vy = self.lateral_velocity * sub_phase  # Right lateral component
            
            # Launch then ballistic arc
            if sub_phase < 0.3:
                vz = self.launch_velocity * (1.0 - sub_phase / 0.3)
            else:
                vz = -0.5 * (sub_phase - 0.3) / 0.7  # Gradual descent
            
            yaw_rate = self.yaw_rate_magnitude * 0.7  # Unwind left yaw
            roll_rate = 0.0
        
        # Phase 0.4-0.6: Right compression with right yaw
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            vx = self.forward_velocity * (1.0 - 0.3 * sub_phase)  # Decelerate slightly
            vy = 0.1 * sub_phase  # Slight right drift
            vz = self.compression_velocity * np.sin(np.pi * sub_phase)  # Smooth compression
            yaw_rate = self.yaw_rate_magnitude  # Rotate right
            roll_rate = 0.3 * sub_phase  # Slight right roll bias
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            vx = self.forward_velocity * (1.0 + 0.5 * sub_phase)  # Accelerate forward
            vy = -self.lateral_velocity * sub_phase  # Left lateral component
            
            # Launch then ballistic arc
            if sub_phase < 0.3:
                vz = self.launch_velocity * (1.0 - sub_phase / 0.3)
            else:
                vz = -0.5 * (sub_phase - 0.3) / 0.7  # Gradual descent
            
            yaw_rate = -self.yaw_rate_magnitude * 0.7  # Unwind right yaw
            roll_rate = 0.0
        
        # Phase 0.8-1.0: Landing preparation and yaw neutralization
        else:
            sub_phase = (phase - 0.8) / 0.2
            vx = self.forward_velocity * (0.9 + 0.1 * sub_phase)  # Stabilize forward speed
            vy = -0.1 * (1.0 - sub_phase)  # Return lateral to zero
            vz = self.compression_velocity * 0.5 * sub_phase  # Gentle landing compression
            yaw_rate = -self.yaw_rate_magnitude * 0.5  # Continue left neutralization
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
        
        Legs compress/extend synchronously but with load emphasis:
        - Left legs (FL, RL) dominant during left-yaw phases (0.0-0.2)
        - Right legs (FR, RR) dominant during right-yaw phases (0.4-0.6)
        - All legs retract during aerial phases (0.2-0.4, 0.6-0.8)
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_left_leg = leg_name.startswith('FL') or leg_name.startswith('RL')
        is_front_leg = leg_name.startswith('FL') or leg_name.startswith('FR')
        
        # Phase 0.0-0.2: Left compression (left legs dominant)
        if phase < 0.2:
            sub_phase = phase / 0.2
            compression_factor = 1.0 if is_left_leg else 0.7  # Left legs compress more
            foot[2] += self.compression_depth * compression_factor * np.sin(np.pi * sub_phase)
            
            # Slight backward shift during compression
            foot[0] -= 0.03 * sub_phase
        
        # Phase 0.2-0.4: Right-diagonal launch and aerial
        elif phase < 0.4:
            sub_phase = (phase - 0.2) / 0.2
            
            # Explosive extension then aerial retraction
            if sub_phase < 0.25:
                # Rapid extension for launch
                extension = sub_phase / 0.25
                foot[2] -= self.compression_depth * (1.0 - extension)
            else:
                # Retract upward during aerial phase
                retract = (sub_phase - 0.25) / 0.75
                foot[2] -= self.step_height * np.sin(np.pi * retract)
                
                # Forward swing in body frame
                foot[0] += 0.08 * retract
                
                # Right legs extend more to prepare for right-side landing
                if not is_left_leg:
                    foot[0] += 0.03 * retract
        
        # Phase 0.4-0.6: Right compression (right legs dominant)
        elif phase < 0.6:
            sub_phase = (phase - 0.4) / 0.2
            compression_factor = 1.0 if not is_left_leg else 0.7  # Right legs compress more
            foot[2] += self.compression_depth * compression_factor * np.sin(np.pi * sub_phase)
            
            # Slight backward shift during compression
            foot[0] -= 0.03 * sub_phase
        
        # Phase 0.6-0.8: Left-diagonal launch and aerial
        elif phase < 0.8:
            sub_phase = (phase - 0.6) / 0.2
            
            # Explosive extension then aerial retraction
            if sub_phase < 0.25:
                # Rapid extension for launch
                extension = sub_phase / 0.25
                foot[2] -= self.compression_depth * (1.0 - extension)
            else:
                # Retract upward during aerial phase
                retract = (sub_phase - 0.25) / 0.75
                foot[2] -= self.step_height * np.sin(np.pi * retract)
                
                # Forward swing in body frame
                foot[0] += 0.08 * retract
                
                # Left legs extend more to prepare for left-side landing
                if is_left_leg:
                    foot[0] += 0.03 * retract
        
        # Phase 0.8-1.0: Landing preparation
        else:
            sub_phase = (phase - 0.8) / 0.2
            
            # Extend legs for landing contact
            foot[2] -= self.step_height * (1.0 - np.sin(np.pi * 0.5 * (1.0 + sub_phase)))
            
            # Begin compression as contact is established
            foot[2] += self.compression_depth * 0.3 * sub_phase
            
            # Position forward for next cycle
            foot[0] += 0.02 * sub_phase
        
        return foot