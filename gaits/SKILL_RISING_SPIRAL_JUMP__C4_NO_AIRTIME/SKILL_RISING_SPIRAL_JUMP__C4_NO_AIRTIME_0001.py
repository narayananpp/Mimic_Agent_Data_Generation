from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising spiral jump with continuous yaw rotation.
    
    Phases:
    - [0.0, 0.2]: Compression - all legs compress symmetrically
    - [0.2, 0.4]: Launch - explosive extension with upward velocity and yaw initiation
    - [0.4, 0.6]: Aerial spiral extension - legs extend radially in spiral sequence
    - [0.6, 0.8]: Peak extension - maximum height with full leg extension
    - [0.8, 1.0]: Descent retraction - legs retract for synchronized landing
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.5  # Skill cycle frequency (slower for jump trajectory)
        
        # Base foot positions (nominal stance in BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Spiral phase offsets for sequential extension (FL -> FR -> RR -> RL)
        self.spiral_phase_offsets = {
            leg_names[0]: 0.0,    # FL leads
            leg_names[1]: 0.025,  # FR second
            leg_names[2]: 0.05,   # RR third
            leg_names[3]: 0.075,  # RL fourth
        }
        
        # Motion parameters
        self.compression_depth = 0.15       # Vertical compression distance
        self.launch_velocity = 2.5          # Peak upward velocity during launch
        self.max_radial_extension = 0.25    # Maximum radial extension from nominal stance
        self.yaw_rate_max = 3.0             # Maximum yaw rotation rate (rad/s)
        self.peak_height = 0.6              # Target peak height above initial position
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocities according to jump trajectory phases.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        if phase < 0.2:
            # Compression: minimal downward velocity
            vz = -0.5 * (phase / 0.2)
            yaw_rate = 0.0
            
        elif phase < 0.4:
            # Launch: explosive upward velocity, yaw initiation
            local_phase = (phase - 0.2) / 0.2
            vz = self.launch_velocity * np.sin(np.pi * local_phase * 0.5)
            yaw_rate = self.yaw_rate_max * local_phase
            
        elif phase < 0.6:
            # Aerial spiral extension: upward velocity decelerating
            local_phase = (phase - 0.4) / 0.2
            vz = self.launch_velocity * (1.0 - local_phase)
            yaw_rate = self.yaw_rate_max
            
        elif phase < 0.8:
            # Peak extension: transition from upward to downward
            local_phase = (phase - 0.6) / 0.2
            vz = -self.launch_velocity * 0.3 * local_phase
            yaw_rate = self.yaw_rate_max
            
        else:
            # Descent retraction: downward velocity, yaw decay
            local_phase = (phase - 0.8) / 0.2
            vz = -self.launch_velocity * 0.5 * (1.0 - 0.3 * local_phase)
            yaw_rate = self.yaw_rate_max * (1.0 - local_phase)
        
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
        Compute foot position in BODY frame for given leg and phase.
        
        Motion structure:
        - [0.0, 0.2]: Vertical compression toward body center
        - [0.2, 0.4]: Explosive extension downward/outward for launch
        - [0.4, 0.6]: Radial extension in spiral sequence (phase-offset per leg)
        - [0.6, 0.8]: Hold maximum extension
        - [0.8, 1.0]: Radial retraction to nominal stance for landing
        """
        base_foot = self.base_feet_pos_body[leg_name].copy()
        foot = base_foot.copy()
        
        # Determine leg-specific spiral phase offset
        spiral_offset = self.spiral_phase_offsets[leg_name]
        
        if phase < 0.2:
            # Compression phase: retract foot vertically toward body
            compression_progress = phase / 0.2
            foot[2] += self.compression_depth * compression_progress
            
        elif phase < 0.4:
            # Launch phase: explosive extension downward/outward
            local_phase = (phase - 0.2) / 0.2
            # Transition from compressed to slightly extended
            foot[2] = base_foot[2] + self.compression_depth * (1.0 - local_phase) - 0.05 * local_phase
            
        elif phase < 0.6:
            # Aerial spiral extension: radial outward motion with spiral offset
            local_phase = (phase - 0.4) / 0.2
            adjusted_phase = local_phase - spiral_offset / 0.2
            adjusted_phase = np.clip(adjusted_phase, 0.0, 1.0)
            
            # Smooth radial extension using sine curve
            extension_factor = np.sin(np.pi * adjusted_phase * 0.5)
            radial_direction = np.array([base_foot[0], base_foot[1], 0.0])
            radial_distance = np.linalg.norm(radial_direction)
            
            if radial_distance > 1e-6:
                radial_unit = radial_direction / radial_distance
                radial_extension = self.max_radial_extension * extension_factor
                foot[0] = base_foot[0] + radial_unit[0] * radial_extension
                foot[1] = base_foot[1] + radial_unit[1] * radial_extension
            
            # Elevate foot during extension
            foot[2] = base_foot[2] + 0.1 * extension_factor
            
        elif phase < 0.8:
            # Peak extension: maintain maximum radial extension
            radial_direction = np.array([base_foot[0], base_foot[1], 0.0])
            radial_distance = np.linalg.norm(radial_direction)
            
            if radial_distance > 1e-6:
                radial_unit = radial_direction / radial_distance
                foot[0] = base_foot[0] + radial_unit[0] * self.max_radial_extension
                foot[1] = base_foot[1] + radial_unit[1] * self.max_radial_extension
            
            foot[2] = base_foot[2] + 0.1
            
        else:
            # Descent retraction: return to nominal stance for landing
            local_phase = (phase - 0.8) / 0.2
            retraction_progress = local_phase
            
            radial_direction = np.array([base_foot[0], base_foot[1], 0.0])
            radial_distance = np.linalg.norm(radial_direction)
            
            if radial_distance > 1e-6:
                radial_unit = radial_direction / radial_distance
                current_extension = self.max_radial_extension * (1.0 - retraction_progress)
                foot[0] = base_foot[0] + radial_unit[0] * current_extension
                foot[1] = base_foot[1] + radial_unit[1] * current_extension
            
            foot[2] = base_foot[2] + 0.1 * (1.0 - retraction_progress)
        
        return foot