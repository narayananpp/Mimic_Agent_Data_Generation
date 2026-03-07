from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump skill implementation.
    
    The robot performs a vertical jump with continuous yaw rotation while legs 
    extend outward in a timed spiral sequence (FL→FR→RR→RL), reaching peak 
    extension at apex, then retracts legs for landing.
    
    Phase breakdown:
      [0.0, 0.2]: compression - all legs compress symmetrically
      [0.2, 0.4]: launch - explosive extension with yaw initiation
      [0.4, 0.6]: aerial_spiral_extension - sequential radial leg extension
      [0.6, 0.8]: apex_hold - full spiral formation at peak altitude
      [0.8, 1.0]: descent_and_landing - leg retraction and landing prep
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Further reduced compression and extension parameters
        self.compression_depth = 0.05  # Reduced from 0.06 to minimize knee flexion stress
        self.compression_inward_factor = 0.06  # Reduced from 0.08
        
        # Leg-specific radial extension factors to address front knee violations
        # Front legs reach further forward naturally, so use more conservative extension
        self.radial_extension_factors = {
            'FL': 1.15,  # Conservative for front-left
            'FR': 1.15,  # Conservative for front-right
            'RL': 1.30,  # Can extend more as rear legs have shorter forward reach
            'RR': 1.30,  # Can extend more as rear legs have shorter forward reach
        }
        
        # Reduced vertical extension to ease kinematic constraints
        self.vertical_extension = 0.02  # Reduced from 0.05
        
        # Calibrated launch and aerial parameters
        self.launch_vz = 0.5  # Further reduced from 0.6 for more controlled ascent
        self.apex_vz = -0.25
        self.descent_vz = -0.7
        
        # Yaw rotation parameters
        self.yaw_rate_launch = 3.0
        self.yaw_rate_aerial = 2.5
        self.yaw_rate_apex = 2.0
        self.yaw_rate_descent = 1.0
        
        # Slightly delayed and smoothed spiral timing for front legs
        self.spiral_peak_phases = {
            'FL': 0.50,  # Delayed from 0.475 to allow base to stabilize
            'FR': 0.54,  # Delayed from 0.525
            'RR': 0.58,  # Slightly earlier from 0.575 to maintain sequence spacing
            'RL': 0.60,  # Unchanged
        }
        
        # Time initialization
        self.t = 0.0
        
        # Base state (world frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base velocity commands based on phase.
        Velocities calibrated to keep base height within [0.1, 0.68]m envelope.
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Gentle downward velocity with smooth profile
            vz = -0.25 * np.sin(local_phase * np.pi)
            yaw_rate = 0.0
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Moderate upward velocity with aggressive early ramp down
            ramp = np.cos(local_phase * np.pi / 2)
            vz = self.launch_vz * ramp
            yaw_rate = self.yaw_rate_launch
            
        # Phase 3: Aerial spiral extension [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Smooth transition to descent
            vz = 0.15 * (1.0 - local_phase) + self.apex_vz * local_phase
            yaw_rate = self.yaw_rate_aerial
            
        # Phase 4: Apex hold [0.6, 0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Descending velocity
            vz = self.apex_vz + (self.descent_vz - self.apex_vz) * local_phase * 0.6
            yaw_rate = self.yaw_rate_apex
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Controlled descent with smooth deceleration
            vz = self.descent_vz * (1.0 - 0.5 * np.sin(local_phase * np.pi / 2))
            yaw_rate = self.yaw_rate_descent * (1.0 - local_phase)
        
        # Set velocity commands
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
        Compute foot position in body frame based on phase and spiral sequence.
        Uses leg-specific extension factors to avoid front knee joint violations.
        """
        
        # Get base foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        foot = foot_base.copy()
        
        # Identify leg for spiral timing and extension factor
        leg_id = None
        for name in self.leg_names:
            if leg_name.startswith(name):
                leg_id = name
                break
        if leg_id is None:
            leg_id = leg_name
            
        spiral_peak = self.spiral_peak_phases.get(leg_id, 0.5)
        radial_factor = self.radial_extension_factors.get(leg_id, 1.2)
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Gentle compression with smooth curve
            compression_curve = np.sin(local_phase * np.pi / 2)
            # Conservative inward pull
            foot[0] *= (1.0 - self.compression_inward_factor * compression_curve)
            foot[1] *= (1.0 - self.compression_inward_factor * compression_curve)
            # Minimal vertical compression
            foot[2] += self.compression_depth * compression_curve
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth transition from compressed to nominal
            compression_curve = np.cos(local_phase * np.pi / 2)
            foot[0] *= (1.0 - self.compression_inward_factor * compression_curve)
            foot[1] *= (1.0 - self.compression_inward_factor * compression_curve)
            foot[2] += self.compression_depth * compression_curve
            # Very gradual liftoff
            if local_phase > 0.7:
                liftoff = (local_phase - 0.7) / 0.3
                liftoff_curve = np.sin(liftoff * np.pi / 2)
                foot[2] -= 0.03 * liftoff_curve
                
        # Phase 3: Aerial spiral extension [0.4, 0.6]
        elif phase < 0.6:
            # Each leg extends radially based on its spiral timing
            if phase < spiral_peak:
                extension_start = 0.4
                extension_progress = (phase - extension_start) / (spiral_peak - extension_start)
                extension_progress = np.clip(extension_progress, 0.0, 1.0)
            else:
                extension_progress = 1.0
                
            # Smooth extension curve with reduced exponent for gentler acceleration
            extension_amount = np.sin(extension_progress * np.pi / 2)
            
            # Leg-specific radial extension in x-y plane
            foot[0] = foot_base[0] * (1.0 + (radial_factor - 1.0) * extension_amount)
            foot[1] = foot_base[1] * (1.0 + (radial_factor - 1.0) * extension_amount)
            
            # Minimal vertical lift during extension
            foot[2] = foot_base[2] - self.vertical_extension * extension_amount
            
        # Phase 4: Apex hold [0.6, 0.8]
        elif phase < 0.8:
            # Maintain full spiral extension with leg-specific factors
            foot[0] = foot_base[0] * radial_factor
            foot[1] = foot_base[1] * radial_factor
            foot[2] = foot_base[2] - self.vertical_extension
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth retraction with ease-in-out curve
            retraction_curve = 0.5 * (1.0 - np.cos(local_phase * np.pi))
            
            # Interpolate from extended to nominal
            extended_x = foot_base[0] * radial_factor
            extended_y = foot_base[1] * radial_factor
            extended_z = foot_base[2] - self.vertical_extension
            
            foot[0] = extended_x + (foot_base[0] - extended_x) * retraction_curve
            foot[1] = extended_y + (foot_base[1] - extended_y) * retraction_curve
            foot[2] = extended_z + (foot_base[2] - extended_z) * retraction_curve
        
        return foot