from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    spiral leg extension pattern during flight phase.
    
    Motion phases:
    - [0.0, 0.2]: Compression prep - all legs compress symmetrically
    - [0.2, 0.4]: Explosive launch - upward velocity generated, yaw rotation starts
    - [0.4, 0.6]: Aerial spiral expansion - legs extend sequentially FL→FR→RR→RL
    - [0.6, 0.8]: Apex full extension - peak height, maximum leg extension
    - [0.8, 1.0]: Descent and landing - legs retract, controlled landing
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters
        self.compression_depth = 0.15  # Vertical compression during prep phase
        self.launch_velocity = 2.5     # Upward velocity magnitude during launch
        self.yaw_rate = 3.0            # Angular velocity around yaw axis (rad/s)
        self.spiral_extension_radius = 0.25  # Radial extension during spiral
        self.spiral_height_offset = 0.1      # Additional height variation in spiral
        
        # Sequential spiral timing offsets for FL→FR→RR→RL pattern
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL starts first
            leg_names[1]: 0.05,  # FR follows
            leg_names[2]: 0.10,  # RR follows
            leg_names[3]: 0.15,  # RL follows last
        }
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Track launch reference for trajectory
        self.launch_height = 0.0
        self.current_vertical_velocity = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation through all phases of the jump.
        """
        vx, vy, vz = 0.0, 0.0, 0.0
        roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0
        
        # Phase 1: Compression prep [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth downward motion during compression
            vz = -self.compression_depth * 2.0 * np.sin(np.pi * local_phase)
            
        # Phase 2: Explosive launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Rapid upward acceleration during launch
            vz = self.launch_velocity * (1.0 + 0.5 * (1.0 - local_phase))
            # Initiate yaw rotation
            yaw_rate = self.yaw_rate
            
        # Phase 3: Aerial spiral expansion [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Decelerating upward velocity (gravity effect)
            vz = self.launch_velocity * (1.0 - 1.5 * local_phase)
            # Constant yaw rotation
            yaw_rate = self.yaw_rate
            
        # Phase 4: Apex full extension [0.6, 0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Transition through apex: upward → downward
            vz = self.launch_velocity * (0.25 - 1.5 * local_phase)
            # Maintain yaw rotation
            yaw_rate = self.yaw_rate
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Downward motion with deceleration near landing
            vz = -self.launch_velocity * (0.5 + 0.5 * local_phase)
            # Decelerate yaw rotation for stable landing
            yaw_rate = self.yaw_rate * (1.0 - local_phase)
        
        # Set velocity commands
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([roll_rate, pitch_rate, yaw_rate])
        
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
        Compute foot position in body frame for each phase of the spiral jump.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine which leg for spiral sequencing
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        # Phase 1: Compression prep [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Legs compress: feet stay at nominal position, vertical compression
            compression_amount = self.compression_depth * np.sin(np.pi * local_phase)
            foot[2] += compression_amount
            
        # Phase 2: Explosive launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Legs extend during launch, feet push down then lift off
            extension = self.compression_depth * (1.0 - local_phase)
            foot[2] += extension
            
        # Phase 3: Aerial spiral expansion [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Sequential spiral extension with phase offset
            adjusted_phase = np.clip(local_phase - spiral_offset / 0.2, 0.0, 1.0)
            spiral_progress = adjusted_phase
            
            # Radial extension in horizontal plane
            direction = base_pos[:2] / (np.linalg.norm(base_pos[:2]) + 1e-6)
            radial_extension = self.spiral_extension_radius * spiral_progress
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            
            # Slight upward component for aesthetic
            foot[2] = base_pos[2] + self.spiral_height_offset * np.sin(np.pi * spiral_progress)
            
        # Phase 4: Apex full extension [0.6, 0.8]
        elif phase < 0.8:
            # Maintain maximum spiral extension
            direction = base_pos[:2] / (np.linalg.norm(base_pos[:2]) + 1e-6)
            radial_extension = self.spiral_extension_radius
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            foot[2] = base_pos[2] + self.spiral_height_offset
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Retract legs back to nominal position for landing
            direction = base_pos[:2] / (np.linalg.norm(base_pos[:2]) + 1e-6)
            radial_extension = self.spiral_extension_radius * (1.0 - local_phase)
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            
            # Lower to landing position with compression
            height_progress = local_phase
            foot[2] = base_pos[2] + self.spiral_height_offset * (1.0 - height_progress)
            # Add landing compression at end
            if local_phase > 0.7:
                landing_compression = self.compression_depth * 0.5 * ((local_phase - 0.7) / 0.3)
                foot[2] += landing_compression
        
        return foot