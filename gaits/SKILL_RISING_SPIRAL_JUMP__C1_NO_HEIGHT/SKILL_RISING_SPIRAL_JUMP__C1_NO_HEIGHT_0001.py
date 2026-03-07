from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    radial leg extension creating a spiral pattern during aerial phase.
    
    Phase breakdown:
      [0.0, 0.2]: Crouch preparation - all legs compress symmetrically
      [0.2, 0.4]: Explosive launch - legs extend, upward velocity, yaw begins
      [0.4, 0.6]: Aerial spiral expansion - legs extend sequentially (FL→FR→RR→RL)
      [0.6, 0.8]: Peak extension - maximum height and radial extension
      [0.8, 1.0]: Descent and landing - legs retract, controlled touchdown
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Slightly slower to allow full jump cycle
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Crouch and extension parameters
        self.crouch_depth = 0.12  # How much legs compress during crouch
        self.max_radial_extension = 0.25  # Maximum outward extension during spiral
        self.aerial_lift_height = 0.05  # Additional z-lift during aerial phase
        
        # Launch parameters
        self.launch_vz_max = 2.5  # Peak vertical velocity during launch
        
        # Yaw rotation parameters
        self.yaw_rate_max = 3.0  # rad/s during main rotation
        
        # Sequential spiral timing (phase offsets for radial extension start)
        self.spiral_extension_start = {
            'FL': 0.40,
            'FR': 0.45,
            'RR': 0.50,
            'RL': 0.55,
        }
        
        # Time tracking
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity state
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with vertical trajectory and yaw rotation.
        
        Vertical velocity profile:
          [0.0-0.2]: Slight downward during crouch
          [0.2-0.4]: Rapid upward acceleration
          [0.4-0.7]: Upward deceleration to zero (ballistic peak)
          [0.7-0.8]: Downward acceleration
          [0.8-1.0]: Downward deceleration for landing
          
        Yaw rate profile:
          [0.0-0.2]: Zero
          [0.2-0.8]: Constant positive rotation
          [0.8-1.0]: Decelerate to zero for landing
        """
        
        # Vertical velocity computation
        if phase < 0.2:
            # Crouch phase: slight downward
            vz = -0.3 * np.sin(np.pi * phase / 0.2)
        elif phase < 0.4:
            # Launch phase: rapid upward acceleration
            progress = (phase - 0.2) / 0.2
            vz = self.launch_vz_max * np.sin(np.pi * progress)
        elif phase < 0.7:
            # Ascending to peak: decelerate upward velocity
            progress = (phase - 0.4) / 0.3
            vz = self.launch_vz_max * 0.6 * (1.0 - progress)
        elif phase < 0.8:
            # Past peak: downward acceleration
            progress = (phase - 0.7) / 0.1
            vz = -self.launch_vz_max * 0.4 * progress
        else:
            # Landing phase: decelerate downward velocity
            progress = (phase - 0.8) / 0.2
            vz = -self.launch_vz_max * 0.4 * (1.0 - progress)
        
        # Yaw rate computation
        if phase < 0.2:
            yaw_rate = 0.0
        elif phase < 0.4:
            # Ramp up yaw rotation during launch
            progress = (phase - 0.2) / 0.2
            yaw_rate = self.yaw_rate_max * progress
        elif phase < 0.8:
            # Maintain constant yaw rotation during aerial phase
            yaw_rate = self.yaw_rate_max
        else:
            # Decelerate yaw for stable landing
            progress = (phase - 0.8) / 0.2
            yaw_rate = self.yaw_rate_max * (1.0 - progress)
        
        # Set velocity commands (world frame)
        self.vel_world = np.array([0.0, 0.0, vz])
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
        Compute foot position in body frame for given leg and phase.
        
        Phases:
          [0.0-0.2]: Crouch - foot retracts inward/upward
          [0.2-0.35]: Launch - foot extends downward (still in contact)
          [0.35-0.8]: Aerial - sequential radial extension spiral, then hold
          [0.8-1.0]: Retract for landing
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific spiral extension start phase
        if leg_name.startswith('FL'):
            spiral_start = self.spiral_extension_start['FL']
        elif leg_name.startswith('FR'):
            spiral_start = self.spiral_extension_start['FR']
        elif leg_name.startswith('RR'):
            spiral_start = self.spiral_extension_start['RR']
        elif leg_name.startswith('RL'):
            spiral_start = self.spiral_extension_start['RL']
        else:
            spiral_start = 0.40
        
        # Phase 1: Crouch preparation [0.0, 0.2]
        if phase < 0.2:
            progress = phase / 0.2
            # Compress legs: retract inward and upward
            crouch_factor = np.sin(np.pi * progress / 2.0)
            foot[2] += self.crouch_depth * crouch_factor
            
        # Phase 2: Launch [0.2, 0.35]
        elif phase < 0.35:
            progress = (phase - 0.2) / 0.15
            # Extend downward for launch
            extension_factor = np.sin(np.pi * progress / 2.0)
            foot[2] -= self.crouch_depth * (1.0 - extension_factor)
            
        # Phase 3: Aerial - spiral expansion and peak [0.35, 0.8]
        elif phase < 0.8:
            # Determine radial extension based on leg-specific timing
            if phase < spiral_start:
                # Not yet extending
                radial_extension = 0.0
            elif phase < spiral_start + 0.2:
                # Extending radially outward
                progress = (phase - spiral_start) / 0.2
                radial_extension = self.max_radial_extension * np.sin(np.pi * progress / 2.0)
            else:
                # Maximum extension maintained
                radial_extension = self.max_radial_extension
            
            # Apply radial extension in horizontal plane
            base_radius = np.sqrt(foot[0]**2 + foot[1]**2)
            if base_radius > 1e-6:
                extension_factor = (base_radius + radial_extension) / base_radius
                foot[0] *= extension_factor
                foot[1] *= extension_factor
            
            # Lift feet slightly during aerial phase for clearance
            aerial_progress = (phase - 0.35) / 0.45
            lift = self.aerial_lift_height * np.sin(np.pi * aerial_progress)
            foot[2] += lift
            
        # Phase 4: Descent and landing [0.8, 1.0]
        else:
            progress = (phase - 0.8) / 0.2
            # Retract from extended position back to nominal
            retraction_factor = 1.0 - progress
            
            # Compute extended position
            base_radius = np.sqrt(self.base_feet_pos_body[leg_name][0]**2 + 
                                self.base_feet_pos_body[leg_name][1]**2)
            if base_radius > 1e-6:
                extension_factor = (base_radius + self.max_radial_extension * retraction_factor) / base_radius
                foot[0] = self.base_feet_pos_body[leg_name][0] * extension_factor
                foot[1] = self.base_feet_pos_body[leg_name][1] * extension_factor
            
            # Lower feet for landing with slight compliance
            foot[2] = self.base_feet_pos_body[leg_name][2] + self.aerial_lift_height * retraction_factor * 0.3
        
        return foot