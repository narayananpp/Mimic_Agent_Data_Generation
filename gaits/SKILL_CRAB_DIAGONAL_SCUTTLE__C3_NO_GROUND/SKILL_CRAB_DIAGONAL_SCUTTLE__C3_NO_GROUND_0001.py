from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab diagonal scuttle gait with constant sideways body orientation.
    
    Motion consists of:
    - Two scuttle strokes (front legs sweep backward, rear legs sweep forward)
    - Rapid reset phase with all legs airborne
    - Glide stabilization phase
    - Body maintains ~90° yaw throughout via zero yaw rate
    - Diagonal motion from simultaneous forward and lateral velocities
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.8  # Cycle frequency
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Scuttle motion parameters
        self.first_stroke_amplitude = 0.12  # First stroke leg sweep distance
        self.second_stroke_amplitude = 0.18  # Second stroke amplified (1.5x)
        self.lift_height = 0.06  # Vertical clearance during reset
        
        # Base velocity parameters
        self.first_stroke_vx = 0.4  # Forward velocity during first stroke
        self.first_stroke_vy = 0.4  # Lateral velocity during first stroke
        self.second_stroke_vx = 0.6  # Amplified forward velocity
        self.second_stroke_vy = 0.6  # Amplified lateral velocity
        self.hop_height = 0.03  # Small vertical hop during reset
        
        # Initial body orientation (90° yaw for crab stance)
        self.initial_yaw = np.pi / 2.0
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = euler_to_quat(0.0, 0.0, self.initial_yaw)
        
    def reset(self, root_pos, root_quat):
        """Override reset to maintain sideways orientation."""
        self.root_pos = root_pos.copy()
        # Force sideways orientation
        self.root_quat = euler_to_quat(0.0, 0.0, self.initial_yaw)
        self.t = 0.0

    def update_base_motion(self, phase, dt):
        """
        Update base using phase-dependent velocities.
        Yaw rate always zero to maintain perpendicular body orientation.
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # Phase 0.0-0.3: First scuttle stroke
        if phase < 0.3:
            vx = self.first_stroke_vx
            vy = self.first_stroke_vy
            
        # Phase 0.3-0.5: Rapid reset with small hop
        elif phase < 0.5:
            local_phase = (phase - 0.3) / 0.2
            # Decay horizontal velocities
            vx = self.first_stroke_vx * (1.0 - local_phase) * 0.5
            vy = self.first_stroke_vy * (1.0 - local_phase) * 0.5
            # Small vertical oscillation
            vz = self.hop_height * np.sin(np.pi * local_phase) * 10.0
            
        # Phase 0.5-0.8: Second amplified stroke
        elif phase < 0.8:
            vx = self.second_stroke_vx
            vy = self.second_stroke_vy
            
        # Phase 0.8-1.0: Glide and stabilize
        else:
            local_phase = (phase - 0.8) / 0.2
            # Smooth decay to zero
            decay = 1.0 - local_phase
            vx = self.second_stroke_vx * decay
            vy = self.second_stroke_vy * decay
        
        # Angular velocities always zero to maintain constant yaw
        self.vel_world = np.array([vx, vy, vz])
        self.omega_world = np.array([0.0, 0.0, 0.0])
        
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
        Compute foot position for each leg based on phase.
        Front legs (FL, FR) sweep backward during strokes.
        Rear legs (RL, RR) sweep forward during strokes.
        All legs lift together during reset phase.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front = leg_name.startswith('F')
        
        # Phase 0.0-0.3: First scuttle stroke
        if phase < 0.3:
            local_phase = phase / 0.3
            if is_front:
                # Front legs sweep backward (negative x)
                foot[0] -= self.first_stroke_amplitude * local_phase
            else:
                # Rear legs sweep forward (positive x)
                foot[0] += self.first_stroke_amplitude * local_phase
                
        # Phase 0.3-0.5: Rapid reset with lift
        elif phase < 0.5:
            local_phase = (phase - 0.3) / 0.2
            # All legs lift and return to start
            if is_front:
                # Return forward
                sweep_offset = -self.first_stroke_amplitude
                foot[0] += sweep_offset * (1.0 - local_phase)
            else:
                # Return backward
                sweep_offset = self.first_stroke_amplitude
                foot[0] += sweep_offset * (1.0 - local_phase)
            # Parabolic lift
            foot[2] += self.lift_height * np.sin(np.pi * local_phase)
            
        # Phase 0.5-0.8: Second amplified stroke
        elif phase < 0.8:
            local_phase = (phase - 0.5) / 0.3
            if is_front:
                # Front legs sweep backward with greater amplitude
                foot[0] -= self.second_stroke_amplitude * local_phase
            else:
                # Rear legs sweep forward with greater amplitude
                foot[0] += self.second_stroke_amplitude * local_phase
                
        # Phase 0.8-1.0: Glide and stabilize
        else:
            local_phase = (phase - 0.8) / 0.2
            # Return to neutral position smoothly
            if is_front:
                sweep_offset = -self.second_stroke_amplitude
                foot[0] += sweep_offset * local_phase
            else:
                sweep_offset = self.second_stroke_amplitude
                foot[0] += sweep_offset * local_phase
        
        return foot