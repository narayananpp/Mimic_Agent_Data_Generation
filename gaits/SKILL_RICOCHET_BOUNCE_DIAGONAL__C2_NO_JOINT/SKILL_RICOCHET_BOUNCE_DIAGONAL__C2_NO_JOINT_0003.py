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
    
    Fix: Properly adjust base foot positions during initialization to ensure
    stance feet maintain ground contact without penetration.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8
        
        # Store original base foot positions and adjust for proper ground clearance
        self.base_feet_pos_body = {}
        for leg_name, pos in initial_foot_positions_body.items():
            adjusted_pos = pos.copy()
            # Raise all foot positions to ensure adequate clearance for stance
            # Original positions are typically below base COM, we add offset to ensure
            # that stance foot Z-coordinate will be at proper ground level
            adjusted_pos[2] = adjusted_pos[2] + 0.10
            self.base_feet_pos_body[leg_name] = adjusted_pos
        
        # Motion parameters
        self.swing_height = 0.10
        self.lateral_retraction = 0.05
        self.forward_reach = 0.03
        
        # Base velocity parameters
        self.vx_compression = 0.4
        self.vx_launch = 1.0
        self.vy_diagonal = 0.45
        self.vz_compression = -0.5
        self.vz_launch = 0.75
        
        # Yaw parameters
        self.yaw_rate_compression = 1.8
        
        # Base state - start at nominal height
        self.t = 0.0
        self.root_pos = np.zeros(3)
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
            vx = self.vx_launch * (1.0 - 0.4 * smooth_phase)
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
            vx = self.vx_launch * (1.0 - 0.4 * smooth_phase)
            vy = -self.vy_diagonal * (1.0 - local_phase)
            vz = self.vz_compression * np.sin(np.pi * smooth_phase) * 0.4
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
        
        During stance phases, feet remain at their base position (adjusted to ensure ground contact).
        During swing phases, feet lift and reposition using additive offsets.
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
                # Keep foot at base position during stance (no additional offset)
                pass
            
            # Phase [0.2-0.4]: Swing - lift during right launch
            elif 0.2 <= phase < 0.4:
                local_phase = (phase - 0.2) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Lift foot off ground with smooth trajectory
                foot[2] += self.swing_height * np.sin(np.pi * local_phase)
                # Forward/backward reach
                if is_front_left:
                    foot[0] += self.forward_reach * smooth_phase
                else:
                    foot[0] -= self.forward_reach * smooth_phase
                # Lateral retraction
                foot[1] += self.lateral_retraction * smooth_phase
            
            # Phase [0.4-0.6]: Swing - maintain elevation during right compression
            elif 0.4 <= phase < 0.6:
                local_phase = (phase - 0.4) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Maintain elevated position with smooth transition
                foot[2] += self.swing_height * (0.5 + 0.5 * np.cos(np.pi * local_phase))
                # Gradually return to neutral
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
                # Smooth transition from swing to stance with gradual descent
                foot[2] += self.swing_height * np.sin(np.pi * (0.5 + 0.5 * local_phase))
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
                # Descend from previous swing cycle
                foot[2] += self.swing_height * np.sin(np.pi * (0.5 + 0.5 * local_phase))
                # Lateral retraction
                foot[1] -= self.lateral_retraction * (1.0 - smooth_phase)
                # Forward/backward return
                if is_front_right:
                    foot[0] += self.forward_reach * (1.0 - smooth_phase)
                else:
                    foot[0] -= self.forward_reach * (1.0 - smooth_phase)
            
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
                # Smooth touchdown transition at beginning of stance
                if local_phase < 0.5:
                    foot[2] += self.swing_height * (1.0 - 2.0 * local_phase)
                # Then maintain stance position
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
            
            # Phase [0.8-1.0]: Swing - maintain elevation for next cycle
            else:
                local_phase = (phase - 0.8) / 0.2
                smooth_phase = smooth_step(local_phase)
                # Maintain elevated position with smooth transition
                foot[2] += self.swing_height * (0.5 + 0.5 * np.cos(np.pi * local_phase))
                # Return motion
                if is_front_right:
                    foot[0] += self.forward_reach * (1.0 - smooth_phase)
                else:
                    foot[0] -= self.forward_reach * (1.0 - smooth_phase)
                foot[1] -= self.lateral_retraction * (1.0 - smooth_phase)
        
        return foot