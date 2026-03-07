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
        
        # Motion parameters - tuned to respect height envelope [0.1, 0.68]
        self.compression_depth = 0.10  # Vertical compression during prep
        self.launch_velocity = 0.9     # Reduced to keep base height within limits
        self.yaw_rate = 3.0            # Angular velocity around yaw axis (rad/s)
        self.spiral_extension_radius = 0.20  # Radial extension during spiral
        self.spiral_height_offset = 0.08     # Height variation in spiral
        
        # Sequential spiral timing offsets for FL→FR→RR→RL pattern
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL starts first
            leg_names[1]: 0.05,  # FR follows
            leg_names[2]: 0.10,  # RR follows
            leg_names[3]: 0.15,  # RL follows last
        }
        
        # Base state - initialize at proper standing height
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, 0.50])  # Start at nominal standing height
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def smooth_transition(self, x):
        """Smoothstep function for continuous transitions"""
        return x * x * (3.0 - 2.0 * x)

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation through all phases of the jump.
        Height-regulated to stay within [0.1, 0.68] envelope.
        """
        vx, vy, vz = 0.0, 0.0, 0.0
        roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0
        
        # Phase 1: Compression prep [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Smooth downward motion during compression using sinusoidal profile
            compression_velocity = -self.compression_depth * np.pi * np.cos(np.pi * local_phase)
            vz = compression_velocity
            
        # Phase 2: Explosive launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            smooth_phase = self.smooth_transition(local_phase)
            # Rapid upward acceleration during launch with smooth ramp-up and ramp-down
            vz = self.launch_velocity * np.sin(np.pi * smooth_phase)
            # Initiate yaw rotation smoothly
            yaw_rate = self.yaw_rate * smooth_phase
            
        # Phase 3: Aerial spiral expansion [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Decelerating upward velocity with aggressive deceleration (factor 2.5)
            vz = self.launch_velocity * (1.0 - 2.5 * local_phase)
            # Constant yaw rotation
            yaw_rate = self.yaw_rate
            
        # Phase 4: Apex full extension [0.6, 0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Smooth transition through apex: upward → downward
            apex_velocity_factor = -0.75 * local_phase - 0.375
            vz = self.launch_velocity * apex_velocity_factor
            # Maintain yaw rotation
            yaw_rate = self.yaw_rate
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth_phase = self.smooth_transition(local_phase)
            # Controlled downward motion with deceleration near landing
            vz = -self.launch_velocity * (0.6 + 0.4 * (1.0 - smooth_phase))
            # Decelerate yaw rotation for stable landing
            yaw_rate = self.yaw_rate * (1.0 - smooth_phase)
        
        # Height safety regulation: dampen upward velocity if approaching ceiling
        height_margin_top = 0.65  # Start damping before hitting 0.68m limit
        height_margin_bottom = 0.15
        
        if self.root_pos[2] > height_margin_top and vz > 0:
            damping_factor = max(0.0, 1.0 - (self.root_pos[2] - height_margin_top) / 0.03)
            vz *= damping_factor
        elif self.root_pos[2] < height_margin_bottom and vz < 0:
            damping_factor = max(0.0, 1.0 - (height_margin_bottom - self.root_pos[2]) / 0.05)
            vz *= damping_factor
        
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
        REVERTED TO ITERATION 1 LOGIC: Feet positions adjusted relative to nominal
        body-frame positions, with base motion handling world-frame height changes.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        # Determine spiral sequencing offset for this leg
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        # Phase 1: Compression prep [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Legs compress: small upward shift in body frame as joints flex
            smooth_compression = self.smooth_transition(local_phase)
            compression_amount = self.compression_depth * np.sin(np.pi * smooth_compression)
            foot[2] += compression_amount
            
        # Phase 2: Explosive launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Legs extend during launch, transitioning from compressed to neutral
            smooth_phase = self.smooth_transition(1.0 - local_phase)
            extension = self.compression_depth * smooth_phase
            foot[2] += extension
            
        # Phase 3: Aerial spiral expansion [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Sequential spiral extension with phase offset
            adjusted_phase = np.clip(local_phase - spiral_offset / 0.2, 0.0, 1.0)
            spiral_progress = self.smooth_transition(adjusted_phase)
            
            # Radial extension in horizontal plane
            horizontal_norm = np.linalg.norm(base_pos[:2])
            if horizontal_norm > 1e-6:
                direction = base_pos[:2] / horizontal_norm
            else:
                direction = np.array([1.0, 0.0])
            
            radial_extension = self.spiral_extension_radius * spiral_progress
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            
            # Smooth upward component for aesthetic spiral
            foot[2] = base_pos[2] + self.spiral_height_offset * np.sin(np.pi * spiral_progress * 0.5)
            
        # Phase 4: Apex full extension [0.6, 0.8]
        elif phase < 0.8:
            # Maintain maximum spiral extension at apex
            horizontal_norm = np.linalg.norm(base_pos[:2])
            if horizontal_norm > 1e-6:
                direction = base_pos[:2] / horizontal_norm
            else:
                direction = np.array([1.0, 0.0])
            
            radial_extension = self.spiral_extension_radius
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            foot[2] = base_pos[2] + self.spiral_height_offset * 0.5
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth_retract = self.smooth_transition(local_phase)
            
            # Retract legs back to nominal position for landing
            horizontal_norm = np.linalg.norm(base_pos[:2])
            if horizontal_norm > 1e-6:
                direction = base_pos[:2] / horizontal_norm
            else:
                direction = np.array([1.0, 0.0])
            
            # Retract horizontal spiral extension
            radial_extension = self.spiral_extension_radius * (1.0 - smooth_retract)
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            
            # Smooth height transition: from spiral offset to landing compression
            height_progress = smooth_retract
            remaining_spiral = self.spiral_height_offset * 0.5 * (1.0 - height_progress)
            
            # Add small landing compression for impact absorption
            if local_phase > 0.6:
                landing_progress = (local_phase - 0.6) / 0.4
                smooth_landing = self.smooth_transition(landing_progress)
                landing_compression = self.compression_depth * 0.4 * smooth_landing
                foot[2] = base_pos[2] + remaining_spiral + landing_compression
            else:
                foot[2] = base_pos[2] + remaining_spiral
        
        return foot