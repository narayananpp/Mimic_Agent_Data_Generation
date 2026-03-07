from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_RISING_SPIRAL_JUMP_MotionGenerator(BaseMotionGenerator):
    """
    Rising Spiral Jump: Vertical jump with continuous yaw rotation and sequential
    radial leg extension creating a spiral pattern during aerial phase.
    
    Phase breakdown:
      [0.0, 0.2]: Preparation - legs gather slightly inward (horizontal retraction)
      [0.2, 0.4]: Explosive launch - legs extend, upward velocity, yaw begins
      [0.4, 0.6]: Aerial spiral expansion - legs extend sequentially (FL→FR→RR→RL)
      [0.6, 0.8]: Peak extension - maximum height and radial extension
      [0.8, 1.0]: Descent and landing - legs retract, controlled touchdown
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters tuned to respect joint limits
        self.max_radial_extension = 0.13  # Proven stable from iteration 2
        self.aerial_lift_height = 0.07  # Increased for robust ground clearance
        self.preparation_retraction = 0.10  # Horizontal gathering instead of vertical crouch
        
        # Launch parameters
        self.launch_vz_max = 2.2
        
        # Yaw rotation parameters
        self.yaw_rate_max = 2.8
        
        # Sequential spiral timing (phase offsets for radial extension start)
        self.spiral_extension_start = {
            'FL': 0.40,
            'FR': 0.45,
            'RR': 0.50,
            'RL': 0.55,
        }
        
        # Time tracking
        self.t = 0.0
        
        # CRITICAL FIX: Compute initial base height from actual foot positions
        # This guarantees feet are at valid ground contact positions initially
        min_foot_z = min(pos[2] for pos in self.base_feet_pos_body.values())
        # Set base height so feet are at z=0.03m in world frame (small safety margin)
        initial_base_height = -min_foot_z + 0.03
        
        self.root_pos = np.array([0.0, 0.0, initial_base_height])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity state
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def update_base_motion(self, phase, dt):
        """
        Update base pose with vertical trajectory and yaw rotation.
        Zero vertical velocity during preparation phase prevents base sinking.
        """
        
        # Vertical velocity computation
        if phase < 0.2:
            # Preparation phase: zero vertical velocity (no base motion)
            vz = 0.0
        elif phase < 0.4:
            # Launch phase: rapid upward acceleration with smooth ramp
            progress = (phase - 0.2) / 0.2
            # Cubic easing for smooth launch
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            vz = self.launch_vz_max * smooth_progress
        elif phase < 0.7:
            # Ascending to peak: decelerate upward velocity
            progress = (phase - 0.4) / 0.3
            vz = self.launch_vz_max * 0.6 * (1.0 - progress)
        elif phase < 0.8:
            # Past peak: downward acceleration
            progress = (phase - 0.7) / 0.1
            vz = -self.launch_vz_max * 0.3 * progress
        else:
            # Landing phase: decelerate to near-zero before touchdown
            progress = (phase - 0.8) / 0.2
            # Smooth deceleration to zero by phase 0.95
            if progress < 0.75:
                vz = -self.launch_vz_max * 0.3 * (1.0 - progress / 0.75)
            else:
                vz = 0.0
        
        # Yaw rate computation
        if phase < 0.2:
            yaw_rate = 0.0
        elif phase < 0.4:
            # Ramp up yaw rotation during launch
            progress = (phase - 0.2) / 0.2
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            yaw_rate = self.yaw_rate_max * smooth_progress
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
        
        CRITICAL FIX: Eliminated vertical crouch motion entirely.
        Preparation phase uses horizontal retraction only to avoid ground penetration.
        Feet maintain nominal z-coordinates during grounded phases.
        """
        
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine leg-specific spiral extension start phase
        spiral_start = self.spiral_extension_start.get(leg_name, 0.40)
        
        # Phase 1: Preparation [0.0, 0.2]
        # Horizontal retraction creates visual "gathering" without vertical motion
        if phase < 0.2:
            progress = phase / 0.2
            # Smooth retraction profile
            retraction_factor = np.sin(np.pi * progress / 2.0)
            # Scale down horizontal components only
            scale = 1.0 - self.preparation_retraction * retraction_factor
            foot[0] *= scale
            foot[1] *= scale
            # Z-coordinate unchanged - feet stay at nominal ground contact height
            
        # Phase 2: Launch [0.2, 0.4]
        elif phase < 0.4:
            progress = (phase - 0.2) / 0.2
            # Return from retracted to nominal horizontal position
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            retraction_factor = 1.0 - smooth_progress
            scale = 1.0 - self.preparation_retraction * retraction_factor
            foot[0] = self.base_feet_pos_body[leg_name][0] * scale
            foot[1] = self.base_feet_pos_body[leg_name][1] * scale
            # Z-coordinate remains at nominal
            
        # Phase 3: Aerial - spiral expansion [0.4, 0.8]
        elif phase < 0.8:
            # Determine radial extension based on leg-specific timing
            if phase < spiral_start:
                radial_extension = 0.0
                lift = self.aerial_lift_height * 0.4  # Early lift for clearance
            elif phase < spiral_start + 0.15:
                # Extending radially outward with smooth curve
                progress = (phase - spiral_start) / 0.15
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                radial_extension = self.max_radial_extension * smooth_progress
                # Gradual lift application
                lift = self.aerial_lift_height * (0.4 + 0.6 * smooth_progress)
            else:
                # Maximum extension maintained
                radial_extension = self.max_radial_extension
                # Full lift at peak
                lift = self.aerial_lift_height
            
            # Apply radial extension in horizontal plane
            base_radius = np.sqrt(foot[0]**2 + foot[1]**2)
            if base_radius > 1e-6:
                extension_factor = (base_radius + radial_extension) / base_radius
                foot[0] *= extension_factor
                foot[1] *= extension_factor
            
            # Apply vertical lift during aerial phase
            foot[2] += lift
            
        # Phase 4: Descent and landing [0.8, 1.0]
        else:
            progress = (phase - 0.8) / 0.2
            # Smooth cubic interpolation back to nominal stance
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            
            # Interpolate horizontal position from extended back to nominal
            base_radius = np.sqrt(self.base_feet_pos_body[leg_name][0]**2 + 
                                self.base_feet_pos_body[leg_name][1]**2)
            
            if base_radius > 1e-6:
                # Compute extended radius at start of landing phase
                extended_radius = base_radius + self.max_radial_extension
                # Interpolate radius
                current_radius = extended_radius * (1.0 - smooth_progress) + base_radius * smooth_progress
                scale = current_radius / base_radius
                foot[0] = self.base_feet_pos_body[leg_name][0] * scale
                foot[1] = self.base_feet_pos_body[leg_name][1] * scale
            else:
                # Direct interpolation for legs near center
                foot[0] = self.base_feet_pos_body[leg_name][0]
                foot[1] = self.base_feet_pos_body[leg_name][1]
            
            # Interpolate vertical position from lifted to nominal ground contact
            start_lift = self.aerial_lift_height
            end_lift = 0.0
            current_lift = start_lift * (1.0 - smooth_progress) + end_lift * smooth_progress
            foot[2] = self.base_feet_pos_body[leg_name][2] + current_lift
        
        return foot