import numpy as np
from utils.math_utils import *
from gaits.base import BaseMotionGenerator


class SKILL_RICOCHET_BOUNCE_DIAGONAL_MotionGenerator(BaseMotionGenerator):
    """
    Ricochet Bounce Diagonal gait: alternating diagonal bounces creating zigzag locomotion.
    
    Motion structure:
    - Phase [0.0-0.2]: Left compression with left yaw (FL+RL stance)
    - Phase [0.2-0.4]: Right launch and flight (all legs airborne)
    - Phase [0.4-0.6]: Right compression with right yaw (FR+RR stance)
    - Phase [0.6-0.8]: Left launch and flight (all legs airborne)
    - Phase [0.8-1.0]: Landing preparation (FL+RL stance)
    
    Critical fix: Stance feet maintain ground contact (Z ≈ constant small positive)
    instead of applying downward offsets that cause ground penetration.
    Compression is achieved via base descent, not foot lowering.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Motion parameters - adjusted for ground clearance
        self.base_height_offset = 0.15  # Raise base to allow compression without ground penetration
        self.stance_foot_z = 0.02  # Small positive Z for ground contact during stance
        self.swing_height = 0.12  # Leg lift during swing phases
        self.lateral_retraction = 0.06  # Lateral leg retraction during swing
        self.forward_reach = 0.04  # Forward/backward reach during swing
        
        # Base velocity parameters - reduced for stability
        self.vx_compression = 0.35
        self.vx_launch = 1.2
        self.vy_diagonal = 0.5
        self.vz_compression = -0.6
        self.vz_launch = 0.9
        
        # Yaw parameters
        self.yaw_rate_compression = 2.0
        
        # Base state
        self.t = 0.0
        self.root_pos = np.array([0.0, 0.0, self.base_height_offset])
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def update_base_motion(self, phase, dt):
        """
        Update base motion with phase-dependent velocities and yaw rates.
        Creates alternating diagonal bounces with yaw modulation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0
        
        # Smoothing function for transitions
        def smooth_step(x):
            return 3*x**2 - 2*x**3
        
        # Phase [0.0-0.2]: Left compression with left yaw
        if 0.0 <= phase < 0.2:
            local_phase = phase / 0.2
            smooth_phase = smooth_step(local_phase)
            vx = self.vx_compression
            vy = 0.0
            vz = self.vz_compression * np.sin(np.pi * smooth_phase)
            yaw_rate = -self.yaw_rate_compression
        
        # Phase [0.2-0.4]: Right launch and flight
        elif 0.2 <= phase < 0.4:
            local_phase = (phase - 0.2) / 0.2
            smooth_phase = smooth_step(local_phase)
            vx = self.vx_compression + (self.vx_launch - self.vx_compression) * smooth_phase
            vy = self.vy_diagonal * np.sin(np.pi * local_phase)
            vz = self.vz_launch * np.sin(np.pi * local_phase)
            yaw_rate = -self.yaw_rate_compression * (1.0 - smooth_phase)
        
        # Phase [0.4-0.6]: Right compression with right yaw
        elif 0.4 <= phase < 0.6:
            local_phase = (phase - 0.4) / 0.2
            smooth_phase = smooth_step(local_phase)
            vx = self.vx_launch * (1.0 - 0.5 * smooth_phase)
            vy = self.vy_diagonal * (1.0 - local_phase)
            vz = self.vz_compression * np.sin(np.pi * smooth_phase)
            yaw_rate = self.yaw_rate_compression
        
        # Phase [0.6-0.8]: Left launch and flight
        elif 0.6 <= phase < 0.8:
            local_phase = (phase - 0.6) / 0.2
            smooth_phase = smooth_step(local_phase)
            vx = self.vx_compression + (self.vx_launch - self.vx_compression) * smooth_phase
            vy = -self.vy_diagonal * np.sin(np.pi * local_phase)
            vz = self.vz_launch * np.sin(np.pi * local_phase)
            yaw_rate = self.yaw_rate_compression * (1.0 - smooth_phase)
        
        # Phase [0.8-1.0]: Landing preparation and neutralization
        else:
            local_phase = (phase - 0.8) / 0.2
            smooth_phase = smooth_step(local_phase)
            vx = self.vx_launch * (1.0 - 0.5 * smooth_phase)
            vy = -self.vy_diagonal * (1.0 - local_phase)
            vz = self.vz_compression * np.sin(np.pi * smooth_phase) * 0.5
            yaw_rate = -self.yaw_rate_compression * smooth_phase
        
        # Set velocities
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
        Compute foot position in body frame for each leg based on phase.
        
        Key fix: During stance phases, feet maintain near-ground Z position.
        Compression is achieved by base descent, not foot lowering.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front_left = leg_name.startswith('FL')
        is_front_right = leg_name.startswith('FR')
        is_rear_left = leg_name.startswith('RL')
        is_rear_right = leg_name.startswith('RR')
        
        # Smoothing function
        def smooth_step(x):
            return 3*x**2 - 2*x**3
        
        # Left diagonal group (FL, RL)
        if is_front_left or is_rear_left:
            # Phase [0.0-0.2]: Stance - left compression
            if 0.0 <= phase < 0.2:
                # Keep foot at ground level during stance
                foot[2] = self.stance_foot_z
            
            # Phase [0.2-0.4]: Swing - lift during right launch
            elif 0.2 <= phase < 0.4:
                local_phase = (phase - 0.2) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Lift foot off ground
                foot[2] += self.swing_height * np.sin(np.pi * local_phase)
                # Forward/backward reach
                if is_front_left:
                    foot[0] += self.forward_reach * smooth_phase
                else:
                    foot[0] -= self.forward_reach * smooth_phase
                # Lateral retraction
                foot[1] += self.lateral_retraction * smooth_phase
            
            # Phase [0.4-0.6]: Swing - retract during right compression
            elif 0.4 <= phase < 0.6:
                local_phase = (phase - 0.4) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Maintain elevated position
                foot[2] += self.swing_height
                # Return to neutral
                if is_front_left:
                    foot[0] += self.forward_reach * (1.0 - smooth_phase)
                else:
                    foot[0] -= self.forward_reach * (1.0 - smooth_phase)
                foot[1] += self.lateral_retraction * (1.0 - smooth_phase)
            
            # Phase [0.6-0.8]: Swing - extend during left launch
            elif 0.6 <= phase < 0.8:
                local_phase = (phase - 0.6) / 0.2
                # Lift foot during flight
                foot[2] += self.swing_height * np.sin(np.pi * local_phase)
                # Forward/backward reach
                if is_front_left:
                    foot[0] += self.forward_reach * local_phase
                else:
                    foot[0] -= self.forward_reach * local_phase
                foot[1] += self.lateral_retraction * local_phase
            
            # Phase [0.8-1.0]: Stance - landing and preparation
            else:
                local_phase = (phase - 0.8) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Smooth transition to ground contact
                foot[2] = self.swing_height * np.sin(np.pi * (0.5 + 0.5 * local_phase)) + self.stance_foot_z
                # Return to neutral position
                if is_front_left:
                    foot[0] += self.forward_reach * (1.0 - smooth_phase)
                else:
                    foot[0] -= self.forward_reach * (1.0 - smooth_phase)
                foot[1] += self.lateral_retraction * (1.0 - smooth_phase)
        
        # Right diagonal group (FR, RR)
        elif is_front_right or is_rear_right:
            # Phase [0.0-0.2]: Swing - retract during left compression
            if 0.0 <= phase < 0.2:
                local_phase = phase / 0.2
                smooth_phase = smooth_step(local_phase)
                # Elevated position from previous cycle
                foot[2] += self.swing_height * (1.0 - local_phase)
                # Lateral retraction
                foot[1] -= self.lateral_retraction * (1.0 - smooth_phase)
            
            # Phase [0.2-0.4]: Swing - extend during right launch
            elif 0.2 <= phase < 0.4:
                local_phase = (phase - 0.2) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Lift during flight
                foot[2] += self.swing_height * np.sin(np.pi * local_phase)
                # Forward/backward reach
                if is_front_right:
                    foot[0] += self.forward_reach * smooth_phase
                else:
                    foot[0] -= self.forward_reach * smooth_phase
                # Lateral motion
                foot[1] -= self.lateral_retraction * smooth_phase
            
            # Phase [0.4-0.6]: Stance - right compression
            elif 0.4 <= phase < 0.6:
                local_phase = (phase - 0.4) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Ground contact during stance - smooth touchdown
                if local_phase < 0.3:
                    foot[2] = self.swing_height * (1.0 - local_phase / 0.3) + self.stance_foot_z
                else:
                    foot[2] = self.stance_foot_z
                # Return to neutral
                if is_front_right:
                    foot[0] += self.forward_reach * (1.0 - smooth_phase)
                else:
                    foot[0] -= self.forward_reach * (1.0 - smooth_phase)
                foot[1] -= self.lateral_retraction * (1.0 - smooth_phase)
            
            # Phase [0.6-0.8]: Swing - extend during left launch
            elif 0.6 <= phase < 0.8:
                local_phase = (phase - 0.6) / 0.2
                # Lift during flight
                foot[2] += self.swing_height * np.sin(np.pi * local_phase)
                # Forward/backward reach
                if is_front_right:
                    foot[0] += self.forward_reach * local_phase
                else:
                    foot[0] -= self.forward_reach * local_phase
                foot[1] -= self.lateral_retraction * local_phase
            
            # Phase [0.8-1.0]: Swing - retract for next cycle
            else:
                local_phase = (phase - 0.8) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Maintain elevated position
                foot[2] += self.swing_height
                # Return motion
                if is_front_right:
                    foot[0] += self.forward_reach * (1.0 - smooth_phase)
                else:
                    foot[0] -= self.forward_reach * (1.0 - smooth_phase)
                foot[1] -= self.lateral_retraction * (1.0 - smooth_phase)
        
        return foot