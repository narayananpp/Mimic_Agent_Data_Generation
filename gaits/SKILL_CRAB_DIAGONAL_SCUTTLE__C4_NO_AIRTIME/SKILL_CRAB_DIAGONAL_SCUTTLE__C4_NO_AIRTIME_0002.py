from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab-style diagonal scuttle gait.
    
    Robot moves diagonally forward-right while body remains oriented sideways.
    All four legs act synchronously in scuttle cycles:
    - Phase 0.0-0.3: First push stroke (all legs grounded)
    - Phase 0.3-0.5: Rapid leg repositioning (all legs airborne)
    - Phase 0.5-0.8: Second amplified push stroke (all legs grounded)
    - Phase 0.8-1.0: Glide and stabilization (all legs grounded)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scuttle motion parameters
        self.stroke_length_first = 0.12  # First push stroke amplitude
        self.stroke_length_second = 0.16  # Second push stroke amplitude (33% larger)
        self.stroke_width = 0.06  # Lateral component of stroke
        self.step_height = 0.12  # Height during swing phase (increased for clearance)
        
        # Base velocity parameters for diagonal motion
        self.vx_first_push = 0.8  # Forward velocity during first push
        self.vy_first_push = 0.8  # Rightward velocity during first push
        self.vx_second_push = 1.1  # Forward velocity during second push (37.5% higher)
        self.vy_second_push = 1.1  # Rightward velocity during second push
        
        # Base height regulation
        self.nominal_base_height = None  # Will be set on first call
        self.height_regulation_gain = 2.0
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity and angular velocity
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)
        
        # All legs synchronized (no phase offsets)
        self.phase_offsets = {leg: 0.0 for leg in leg_names}

    def update_base_motion(self, phase, dt):
        """
        Update base motion to generate diagonal scuttle movement.
        Body yaw remains constant (sideways orientation) throughout.
        Base height is actively regulated to prevent ground penetration.
        """
        # Initialize nominal base height on first call
        if self.nominal_base_height is None:
            self.nominal_base_height = self.root_pos[2]
        
        vx = 0.0
        vy = 0.0
        vz_base = 0.0
        
        # Phase 0.0-0.3: First push stroke
        if 0.0 <= phase < 0.3:
            vx = self.vx_first_push
            vy = self.vy_first_push
            vz_base = 0.0  # No downward press in kinematic system
        
        # Phase 0.3-0.5: Leg reset (coast on momentum with slight lift)
        elif 0.3 <= phase < 0.5:
            progress = (phase - 0.3) / 0.2
            # Decaying velocity during coast
            decay = 1.0 - 0.4 * progress
            vx = self.vx_first_push * decay
            vy = self.vy_first_push * decay
            # Slight upward motion during leg repositioning
            vz_base = 0.15 * np.sin(np.pi * progress)
        
        # Phase 0.5-0.8: Second amplified push stroke
        elif 0.5 <= phase < 0.8:
            vx = self.vx_second_push
            vy = self.vy_second_push
            vz_base = 0.0  # No downward press
        
        # Phase 0.8-1.0: Glide and stabilization
        else:
            progress = (phase - 0.8) / 0.2
            # Smooth decay to zero
            decay = 1.0 - progress
            vx = self.vx_second_push * decay * 0.5
            vy = self.vy_second_push * decay * 0.5
            vz_base = 0.0
        
        # Base height regulation: add corrective vz to maintain nominal height
        height_error = self.nominal_base_height - self.root_pos[2]
        vz_correction = self.height_regulation_gain * height_error
        vz = vz_base + vz_correction
        
        self.vel_world = np.array([vx, vy, vz])
        # Yaw rate is zero throughout to maintain sideways body orientation
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame for synchronized scuttle motion.
        
        Front legs (FL, FR) sweep rearward during push.
        Rear legs (RL, RR) sweep forward during push.
        All legs lift and reposition simultaneously during reset.
        During stance phases, feet are positioned to maintain ground contact.
        """
        leg_phase = (phase + self.phase_offsets[leg_name]) % 1.0
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine if front or rear leg
        is_front = leg_name.startswith('F')
        is_left = leg_name.endswith('L')
        
        # Stance foot height adjustment to ensure ground contact
        # During stance, foot should be positioned to touch ground (z=0 world frame)
        # Assuming nominal base height, foot z in body frame should ensure world z = 0
        stance_z_offset = 0.0  # Will maintain base foot position during stance
        
        # Phase 0.0-0.3: First push stroke (stance)
        if leg_phase < 0.3:
            progress = leg_phase / 0.3
            if is_front:
                # Front legs sweep rearward
                foot[0] -= self.stroke_length_first * (progress - 0.5)
            else:
                # Rear legs sweep forward
                foot[0] += self.stroke_length_first * (progress - 0.5)
            # Slight outward motion for all legs
            if is_left:
                foot[1] -= self.stroke_width * (progress - 0.5)
            else:
                foot[1] += self.stroke_width * (progress - 0.5)
            # Keep foot at stance height (no z modification for ground contact)
            foot[2] = self.base_feet_pos_body[leg_name][2]
        
        # Phase 0.3-0.5: Rapid aerial repositioning (swing)
        elif leg_phase < 0.5:
            progress = (leg_phase - 0.3) / 0.2
            # Smooth swing trajectory
            swing_progress = (1.0 - np.cos(np.pi * progress)) / 2.0
            
            if is_front:
                # Front legs return from rear to forward
                x_start = -self.stroke_length_first * 0.5
                x_end = self.stroke_length_second * 0.5
                foot[0] += x_start + swing_progress * (x_end - x_start)
            else:
                # Rear legs return from forward to rear
                x_start = self.stroke_length_first * 0.5
                x_end = -self.stroke_length_second * 0.5
                foot[0] += x_start + swing_progress * (x_end - x_start)
            
            # Lateral repositioning
            if is_left:
                y_start = -self.stroke_width * 0.5
                y_end = self.stroke_width * 1.3 * 0.5
                foot[1] += y_start + swing_progress * (y_end - y_start)
            else:
                y_start = self.stroke_width * 0.5
                y_end = -self.stroke_width * 1.3 * 0.5
                foot[1] += y_start + swing_progress * (y_end - y_start)
            
            # Lift foot during swing with smooth arc
            foot[2] += self.step_height * np.sin(np.pi * progress)
        
        # Phase 0.5-0.8: Second amplified push stroke (stance)
        elif leg_phase < 0.8:
            progress = (leg_phase - 0.5) / 0.3
            if is_front:
                # Front legs sweep rearward with larger amplitude
                foot[0] -= self.stroke_length_second * (progress - 0.5)
            else:
                # Rear legs sweep forward with larger amplitude
                foot[0] += self.stroke_length_second * (progress - 0.5)
            # Amplified lateral motion
            if is_left:
                foot[1] -= self.stroke_width * 1.3 * (progress - 0.5)
            else:
                foot[1] += self.stroke_width * 1.3 * (progress - 0.5)
            # Keep foot at stance height
            foot[2] = self.base_feet_pos_body[leg_name][2]
        
        # Phase 0.8-1.0: Glide and stabilization (stance)
        else:
            progress = (leg_phase - 0.8) / 0.2
            # Smooth transition to neutral position using cosine interpolation
            transition = (1.0 - np.cos(np.pi * progress)) / 2.0
            
            if is_front:
                x_start = -self.stroke_length_second * 0.5
                foot[0] += x_start * (1.0 - transition)
            else:
                x_start = self.stroke_length_second * 0.5
                foot[0] += x_start * (1.0 - transition)
            
            # Return to neutral lateral position
            if is_left:
                y_start = -self.stroke_width * 1.3 * 0.5
                foot[1] += y_start * (1.0 - transition)
            else:
                y_start = self.stroke_width * 1.3 * 0.5
                foot[1] += y_start * (1.0 - transition)
            
            # Keep foot at stance height
            foot[2] = self.base_feet_pos_body[leg_name][2]
        
        return foot