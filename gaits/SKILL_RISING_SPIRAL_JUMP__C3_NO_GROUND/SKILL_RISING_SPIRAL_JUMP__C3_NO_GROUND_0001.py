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
        self.freq = 0.5  # Slower frequency for dramatic aerial maneuver
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Compression and extension parameters
        self.compression_depth = 0.12  # How much legs compress during crouch
        self.radial_extension_factor = 1.8  # Multiplier for spiral extension distance
        self.vertical_extension = 0.15  # Additional vertical lift during spiral
        
        # Launch and aerial parameters
        self.launch_vz = 2.5  # Strong upward velocity during launch
        self.apex_vz = 0.0  # Near-zero velocity at apex
        self.descent_vz = -2.0  # Descent velocity
        
        # Yaw rotation parameters (target ~270 degrees total rotation)
        self.yaw_rate_launch = 3.0  # rad/s during launch
        self.yaw_rate_aerial = 2.5  # rad/s during aerial phase
        self.yaw_rate_apex = 2.0  # rad/s at apex
        self.yaw_rate_descent = 1.0  # rad/s during descent (decelerating)
        
        # Spiral timing offsets for each leg (when they reach max extension)
        # FL first, then FR, RR, RL in sequence
        self.spiral_peak_phases = {
            'FL': 0.475,  # First to extend
            'FR': 0.525,  # Second
            'RR': 0.575,  # Third
            'RL': 0.600,  # Last to complete spiral
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
        
        Compression: slight downward movement
        Launch: strong upward velocity + yaw initiation
        Aerial: continuing upward with deceleration + steady yaw
        Apex: near-zero vertical velocity + steady yaw
        Descent: downward velocity + yaw deceleration
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Slight downward velocity transitioning to zero
            vz = -0.5 * (1.0 - local_phase)
            yaw_rate = 0.0
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Explosive upward velocity
            vz = self.launch_vz
            # Initiate yaw rotation
            yaw_rate = self.yaw_rate_launch
            
        # Phase 3: Aerial spiral extension [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Decelerating upward velocity
            vz = self.launch_vz * (1.0 - local_phase * 0.7)
            # Steady yaw rotation
            yaw_rate = self.yaw_rate_aerial
            
        # Phase 4: Apex hold [0.6, 0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Near-zero velocity at apex, transitioning to descent
            vz = self.apex_vz - local_phase * 0.5
            # Steady yaw rotation
            yaw_rate = self.yaw_rate_apex
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Descending velocity
            vz = self.descent_vz
            # Decelerating yaw rotation
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
        
        Each leg follows:
        - Compression: retract inward and downward
        - Launch: extend to nominal + push off
        - Spiral extension: radial outward extension in timed sequence
        - Apex hold: maintain extended position
        - Descent: retract inward for landing
        """
        
        # Get base foot position
        foot_base = self.base_feet_pos_body[leg_name].copy()
        foot = foot_base.copy()
        
        # Identify leg for spiral timing
        leg_id = None
        for name in self.leg_names:
            if leg_name.startswith(name):
                leg_id = name
                break
        if leg_id is None:
            leg_id = leg_name
            
        spiral_peak = self.spiral_peak_phases.get(leg_id, 0.5)
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Compress inward and downward
            compression_factor = np.sin(local_phase * np.pi)
            foot[0] *= (1.0 - 0.15 * compression_factor)  # Pull x inward
            foot[1] *= (1.0 - 0.15 * compression_factor)  # Pull y inward
            foot[2] += self.compression_depth * compression_factor  # Lower z
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Extend back toward nominal, then push off
            compression_factor = np.sin((1.0 - local_phase) * np.pi)
            foot[0] *= (1.0 - 0.15 * compression_factor)
            foot[1] *= (1.0 - 0.15 * compression_factor)
            foot[2] += self.compression_depth * compression_factor
            # Lift off ground near end of phase
            if local_phase > 0.7:
                liftoff = (local_phase - 0.7) / 0.3
                foot[2] -= 0.05 * liftoff
                
        # Phase 3: Aerial spiral extension [0.4, 0.6]
        elif phase < 0.6:
            # Each leg extends radially based on its spiral timing
            # Extension peaks at spiral_peak_phase for each leg
            if phase < spiral_peak:
                # Extending phase
                extension_start = 0.4
                extension_progress = (phase - extension_start) / (spiral_peak - extension_start)
                extension_progress = np.clip(extension_progress, 0.0, 1.0)
            else:
                # Past peak, hold extended
                extension_progress = 1.0
                
            # Smooth extension curve
            extension_amount = np.sin(extension_progress * np.pi / 2)
            
            # Radial extension in x-y plane
            foot[0] = foot_base[0] * (1.0 + (self.radial_extension_factor - 1.0) * extension_amount)
            foot[1] = foot_base[1] * (1.0 + (self.radial_extension_factor - 1.0) * extension_amount)
            
            # Vertical lift during extension
            foot[2] = foot_base[2] - self.vertical_extension * extension_amount
            
        # Phase 4: Apex hold [0.6, 0.8]
        elif phase < 0.8:
            # Maintain full spiral extension
            foot[0] = foot_base[0] * self.radial_extension_factor
            foot[1] = foot_base[1] * self.radial_extension_factor
            foot[2] = foot_base[2] - self.vertical_extension
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Retract legs inward for landing
            retraction_progress = local_phase
            retraction_curve = np.sin(retraction_progress * np.pi / 2)
            
            # Interpolate from extended to nominal
            extended_x = foot_base[0] * self.radial_extension_factor
            extended_y = foot_base[1] * self.radial_extension_factor
            extended_z = foot_base[2] - self.vertical_extension
            
            foot[0] = extended_x + (foot_base[0] - extended_x) * retraction_curve
            foot[1] = extended_y + (foot_base[1] - extended_y) * retraction_curve
            foot[2] = extended_z + (foot_base[2] - extended_z) * retraction_curve
            
            # Final landing preparation - feet move toward ground contact
            if local_phase > 0.75:
                landing_prep = (local_phase - 0.75) / 0.25
                foot[2] += 0.03 * landing_prep  # Slight downward reach for contact
        
        return foot