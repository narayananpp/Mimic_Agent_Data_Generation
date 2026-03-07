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
        self.freq = 0.8  # Increased frequency to reduce integration time
        
        # Base foot positions (nominal stance)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Reduced compression and extension parameters to avoid joint limits
        self.compression_depth = 0.06  # Reduced from 0.12 to avoid knee hyperflexion
        self.radial_extension_factor = 1.3  # Reduced from 1.8 to stay within kinematic reach
        self.vertical_extension = 0.05  # Reduced from 0.15 to ease knee extension requirements
        self.compression_inward_factor = 0.08  # Reduced from 0.15
        
        # Reduced launch and aerial parameters to keep base height within envelope [0.1, 0.68]
        self.launch_vz = 0.6  # Reduced from 2.5 to prevent excessive altitude gain
        self.apex_vz = -0.3  # Start descending earlier at apex
        self.descent_vz = -0.8  # Moderate descent velocity
        
        # Yaw rotation parameters (preserve rotational aesthetic)
        self.yaw_rate_launch = 3.0
        self.yaw_rate_aerial = 2.5
        self.yaw_rate_apex = 2.0
        self.yaw_rate_descent = 1.0
        
        # Spiral timing offsets for each leg (preserve spiral sequence)
        self.spiral_peak_phases = {
            'FL': 0.475,
            'FR': 0.525,
            'RR': 0.575,
            'RL': 0.600,
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
        Velocities are calibrated to keep base height within [0.1, 0.68]m envelope.
        """
        
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Phase 1: Compression [0.0, 0.2]
        if phase < 0.2:
            local_phase = phase / 0.2
            # Gentle downward velocity during compression
            vz = -0.3 * np.sin(local_phase * np.pi)
            yaw_rate = 0.0
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Moderate upward velocity with smooth ramp down
            ramp = 1.0 - 0.6 * local_phase
            vz = self.launch_vz * ramp
            yaw_rate = self.yaw_rate_launch
            
        # Phase 3: Aerial spiral extension [0.4, 0.6]
        elif phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            # Transition from small upward to downward velocity
            vz = 0.2 * (1.0 - local_phase) + self.apex_vz * local_phase
            yaw_rate = self.yaw_rate_aerial
            
        # Phase 4: Apex hold [0.6, 0.8]
        elif phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            # Descending velocity starting at apex
            vz = self.apex_vz + (self.descent_vz - self.apex_vz) * local_phase * 0.5
            yaw_rate = self.yaw_rate_apex
            
        # Phase 5: Descent and landing [0.8, 1.0]
        else:
            local_phase = (phase - 0.8) / 0.2
            # Controlled descent with deceleration near landing
            vz = self.descent_vz * (1.0 - 0.4 * local_phase)
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
        Parameters tuned to avoid joint limit violations while preserving spiral aesthetic.
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
            # Gentle compression with smooth sinusoidal curve
            compression_curve = np.sin(local_phase * np.pi)
            # Reduced inward pull to avoid knee hyperflexion
            foot[0] *= (1.0 - self.compression_inward_factor * compression_curve)
            foot[1] *= (1.0 - self.compression_inward_factor * compression_curve)
            # Reduced vertical compression
            foot[2] += self.compression_depth * compression_curve
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            # Smooth transition from compressed to nominal
            compression_curve = np.sin((1.0 - local_phase) * np.pi)
            foot[0] *= (1.0 - self.compression_inward_factor * compression_curve)
            foot[1] *= (1.0 - self.compression_inward_factor * compression_curve)
            foot[2] += self.compression_depth * compression_curve
            # Gradual liftoff near end of phase
            if local_phase > 0.6:
                liftoff = (local_phase - 0.6) / 0.4
                liftoff_curve = liftoff * liftoff
                foot[2] -= 0.04 * liftoff_curve
                
        # Phase 3: Aerial spiral extension [0.4, 0.6]
        elif phase < 0.6:
            # Each leg extends radially based on its spiral timing
            if phase < spiral_peak:
                extension_start = 0.4
                extension_progress = (phase - extension_start) / (spiral_peak - extension_start)
                extension_progress = np.clip(extension_progress, 0.0, 1.0)
            else:
                extension_progress = 1.0
                
            # Smooth extension curve using squared sine for gentle acceleration
            extension_amount = np.sin(extension_progress * np.pi / 2) ** 1.5
            
            # Conservative radial extension in x-y plane
            foot[0] = foot_base[0] * (1.0 + (self.radial_extension_factor - 1.0) * extension_amount)
            foot[1] = foot_base[1] * (1.0 + (self.radial_extension_factor - 1.0) * extension_amount)
            
            # Reduced vertical lift during extension
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
            # Smooth retraction using ease-in curve
            retraction_curve = local_phase ** 1.5
            
            # Interpolate from extended to nominal
            extended_x = foot_base[0] * self.radial_extension_factor
            extended_y = foot_base[1] * self.radial_extension_factor
            extended_z = foot_base[2] - self.vertical_extension
            
            foot[0] = extended_x + (foot_base[0] - extended_x) * retraction_curve
            foot[1] = extended_y + (foot_base[1] - extended_y) * retraction_curve
            foot[2] = extended_z + (foot_base[2] - extended_z) * retraction_curve
        
        return foot