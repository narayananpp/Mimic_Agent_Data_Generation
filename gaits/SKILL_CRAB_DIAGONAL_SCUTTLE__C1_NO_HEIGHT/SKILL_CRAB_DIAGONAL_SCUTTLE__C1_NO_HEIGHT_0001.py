from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_CRAB_DIAGONAL_SCUTTLE_MotionGenerator(BaseMotionGenerator):
    """
    Crab diagonal scuttle gait with perpendicular body orientation.
    
    The robot moves diagonally forward-right while maintaining body orientation
    perpendicular to travel direction. Motion consists of two scuttle strokes
    with coordinated leg sweeping, separated by reset and glide phases.
    
    Phase structure:
    - [0.0, 0.3]: First scuttle stroke (all legs in contact, coordinated sweep)
    - [0.3, 0.5]: Rapid reset (all legs airborne, repositioning)
    - [0.5, 0.8]: Second scuttle stroke (amplified, all legs in contact)
    - [0.8, 1.0]: Glide stabilization (all legs airborne)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Scuttle stroke parameters
        self.sweep_length_front = 0.12  # Front legs sweep rearward
        self.sweep_length_rear = 0.12   # Rear legs sweep forward
        self.step_height = 0.10
        
        # Amplification factor for second stroke
        self.second_stroke_amp = 1.3
        
        # Base foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity parameters for diagonal motion
        self.vx_base = 0.8  # Forward velocity component
        self.vy_base = 0.8  # Rightward velocity component (diagonal ~45 degrees)
        self.vz_reset = 0.15  # Brief upward velocity during reset

    def update_base_motion(self, phase, dt):
        """
        Update base motion with diagonal velocity during scuttle strokes.
        Body orientation remains perpendicular to travel (zero yaw rate).
        """
        vx = 0.0
        vy = 0.0
        vz = 0.0
        
        # First scuttle stroke: diagonal motion
        if 0.0 <= phase < 0.3:
            progress = phase / 0.3
            vx = self.vx_base * (1.0 - 0.3 * progress)  # Slight decay
            vy = self.vy_base * (1.0 - 0.3 * progress)
            
        # Rapid reset: deceleration with brief upward motion
        elif 0.3 <= phase < 0.5:
            progress = (phase - 0.3) / 0.2
            if progress < 0.3:
                vz = self.vz_reset * np.sin(np.pi * progress / 0.3)
            vx = self.vx_base * 0.7 * (1.0 - progress)
            vy = self.vy_base * 0.7 * (1.0 - progress)
            
        # Second scuttle stroke: amplified diagonal motion
        elif 0.5 <= phase < 0.8:
            progress = (phase - 0.5) / 0.3
            vx = self.vx_base * self.second_stroke_amp * (1.0 - 0.3 * progress)
            vy = self.vy_base * self.second_stroke_amp * (1.0 - 0.3 * progress)
            
        # Glide phase: momentum decay
        elif 0.8 <= phase <= 1.0:
            progress = (phase - 0.8) / 0.2
            decay = np.exp(-3.0 * progress)
            vx = self.vx_base * self.second_stroke_amp * 0.7 * decay
            vy = self.vy_base * self.second_stroke_amp * 0.7 * decay
        
        # Maintain perpendicular body orientation (zero yaw rate)
        self.vel_world = np.array([vx, vy, vz])
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
        Compute foot position in body frame based on phase.
        
        Front legs (FL, FR) sweep rearward during scuttle.
        Rear legs (RL, RR) sweep forward during scuttle.
        All legs synchronized during each stroke.
        """
        foot = self.base_feet_pos_body[leg_name].copy()
        
        is_front_leg = leg_name.startswith('F')
        
        # First scuttle stroke: all legs in contact, coordinated sweep
        if 0.0 <= phase < 0.3:
            progress = phase / 0.3
            if is_front_leg:
                # Front legs sweep rearward (negative x in body frame)
                foot[0] -= self.sweep_length_front * (progress - 0.5)
            else:
                # Rear legs sweep forward (positive x in body frame)
                foot[0] += self.sweep_length_rear * (progress - 0.5)
        
        # Rapid reset: all legs airborne, repositioning
        elif 0.3 <= phase < 0.5:
            progress = (phase - 0.3) / 0.2
            
            # Lift legs up
            swing_height = self.step_height * np.sin(np.pi * progress)
            foot[2] += swing_height
            
            if is_front_leg:
                # Front legs return forward for next stroke
                foot[0] -= self.sweep_length_front * (0.5 - progress)
            else:
                # Rear legs return rearward for next stroke
                foot[0] += self.sweep_length_rear * (0.5 - progress)
        
        # Second scuttle stroke: amplified, all legs in contact
        elif 0.5 <= phase < 0.8:
            progress = (phase - 0.5) / 0.3
            amp = self.second_stroke_amp
            
            if is_front_leg:
                # Front legs sweep rearward with amplification
                foot[0] -= self.sweep_length_front * amp * (progress - 0.5)
            else:
                # Rear legs sweep forward with amplification
                foot[0] += self.sweep_length_rear * amp * (progress - 0.5)
        
        # Glide phase: all legs airborne, held in position
        elif 0.8 <= phase <= 1.0:
            progress = (phase - 0.8) / 0.2
            
            # Hold legs slightly elevated
            foot[2] += self.step_height * 0.6 * (1.0 - progress * 0.5)
            
            # Position bias for smooth cycle restart
            if is_front_leg:
                # Front legs biased forward
                foot[0] -= self.sweep_length_front * self.second_stroke_amp * (0.5 - 0.3 * progress)
            else:
                # Rear legs biased rearward
                foot[0] += self.sweep_length_rear * self.second_stroke_amp * (0.5 - 0.3 * progress)
        
        return foot