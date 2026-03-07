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
    
    Strategy: Restore iteration 1 proven foot logic with moderate launch velocity
    and explicit height clamping to satisfy both constraints simultaneously.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - balanced for height compliance and ground contact
        self.compression_depth = 0.14  # Adequate range for compression/landing
        self.launch_velocity = 1.5     # Moderate velocity with height clamping
        self.yaw_rate = 3.0            # Angular velocity around yaw axis (rad/s)
        self.spiral_extension_radius = 0.18  # Conservative radial extension
        self.spiral_height_offset = 0.06     # Modest height variation in spiral
        
        # Sequential spiral timing offsets for FL→FR→RR→RL pattern
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,   # FL starts first
            leg_names[1]: 0.05,  # FR follows
            leg_names[2]: 0.10,  # RR follows
            leg_names[3]: 0.15,  # RL follows last
        }
        
        # Base state
        self.t = 0.0
        self.nominal_standing_height = 0.50
        self.root_pos = np.array([0.0, 0.0, self.nominal_standing_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def smooth_transition(self, x):
        """Smoothstep function for continuous transitions"""
        return x * x * (3.0 - 2.0 * x)

    def update_base_motion(self, phase, dt):
        """
        Update base position and orientation with explicit height clamping.
        """
        vx, vy, vz = 0.0, 0.0, 0.0
        roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0
        
        # Phase 1: Compression prep [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            vz = -self.compression_depth * np.pi * np.cos(np.pi * local_phase)
            
        # Phase 2: Explosive launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            smooth_phase = self.smooth_transition(local_phase)
            vz = self.launch_velocity * np.sin(np.pi * smooth_phase)
            yaw_rate = self.yaw_rate * smooth_phase
            
        # Phase 3: Aerial spiral expansion [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Aggressive deceleration to limit peak height
            vz = self.launch_velocity * (1.0 - 3.5 * local_phase)
            yaw_rate = self.yaw_rate
            
        # Phase 4: Apex full extension [0.6, 0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            vz = -self.launch_velocity * (0.5 + 1.5 * local_phase)
            yaw_rate = self.yaw_rate
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth_phase = self.smooth_transition(local_phase)
            
            # Height-targeted descent for cyclic consistency
            height_error = self.root_pos[2] - self.nominal_standing_height
            base_descent = -self.launch_velocity * 0.8
            
            if local_phase < 0.6:
                vz = base_descent
            else:
                correction = 1.0 - (local_phase - 0.6) / 0.4
                vz = base_descent * correction - height_error * 3.0
            
            yaw_rate = self.yaw_rate * (1.0 - smooth_phase)
        
        # Height safety: dampen velocity near limits
        if self.root_pos[2] > 0.63 and vz > 0:
            damping = max(0.0, 1.0 - (self.root_pos[2] - 0.63) / 0.05)
            vz *= damping
        elif self.root_pos[2] < 0.20 and vz < 0:
            damping = max(0.0, 1.0 - (0.20 - self.root_pos[2]) / 0.10)
            vz *= damping
        
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
        
        # HARD CLAMP: Enforce absolute height limits
        self.root_pos[2] = np.clip(self.root_pos[2], 0.15, 0.67)

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame using iteration 1 proven logic.
        Small positive adjustments during compression/landing matched to base motion.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        foot = base_pos.copy()
        
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        # Phase 1: Compression prep [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            smooth_compression = self.smooth_transition(local_phase)
            compression_amount = self.compression_depth * np.sin(np.pi * smooth_compression)
            foot[2] += compression_amount
            
        # Phase 2: Explosive launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            smooth_phase = self.smooth_transition(1.0 - local_phase)
            extension = self.compression_depth * smooth_phase
            foot[2] += extension
            
        # Phase 3: Aerial spiral expansion [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            adjusted_phase = np.clip(local_phase - spiral_offset / 0.2, 0.0, 1.0)
            spiral_progress = self.smooth_transition(adjusted_phase)
            
            horizontal_norm = np.linalg.norm(base_pos[:2])
            if horizontal_norm > 1e-6:
                direction = base_pos[:2] / horizontal_norm
            else:
                direction = np.array([1.0, 0.0])
            
            radial_extension = self.spiral_extension_radius * spiral_progress
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            foot[2] = base_pos[2] + self.spiral_height_offset * np.sin(np.pi * spiral_progress * 0.5)
            
        # Phase 4: Apex full extension [0.6, 0.8]
        elif phase < 0.8:
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
            
            horizontal_norm = np.linalg.norm(base_pos[:2])
            if horizontal_norm > 1e-6:
                direction = base_pos[:2] / horizontal_norm
            else:
                direction = np.array([1.0, 0.0])
            
            radial_extension = self.spiral_extension_radius * (1.0 - smooth_retract)
            foot[0] = base_pos[0] + direction[0] * radial_extension
            foot[1] = base_pos[1] + direction[1] * radial_extension
            
            # Smooth transition from spiral to landing with compression
            height_progress = smooth_retract
            spiral_remaining = self.spiral_height_offset * 0.5 * (1.0 - height_progress)
            
            # Add landing compression that returns to nominal at phase 1.0
            landing_compression = self.compression_depth * 0.5 * np.sin(np.pi * height_progress)
            
            foot[2] = base_pos[2] + spiral_remaining + landing_compression
        
        return foot