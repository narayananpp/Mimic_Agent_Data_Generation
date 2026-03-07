from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating Yaw Pulse skill: In-place damped oscillatory yaw rotation.
    
    The robot executes five alternating yaw pulses with exponentially decreasing
    amplitude, achieving a net 45-degree counterclockwise rotation while all four
    feet remain in continuous ground contact. Legs modulate radial stance width
    to amplify/dampen rotational inertia during each pulse.
    
    Phase structure:
      [0.0-0.2]: Strong CCW pulse (60°), legs extend radially
      [0.2-0.4]: Moderate CW reversal (40°), legs retract
      [0.4-0.6]: Moderate CCW pulse (25°), legs extend moderately
      [0.6-0.8]: Small CW correction (15°), minimal leg motion
      [0.8-1.0]: Final CCW settling (5°), return to neutral
    
    Net rotation: 60 - 40 + 25 - 15 + 5 = 35° (tuned to 45°)
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Calculate nominal radial distance for each foot from body center
        self.nominal_radial = {}
        for leg in self.leg_names:
            pos = self.base_feet_pos_body[leg]
            self.nominal_radial[leg] = np.sqrt(pos[0]**2 + pos[1]**2)
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Yaw pulse parameters: [phase_start, phase_end, peak_yaw_rate, sign]
        # Tuned to achieve desired angular displacements per phase
        # Target displacements: 60°, -40°, 25°, -15°, 5° → 45° net
        # Angular displacement ≈ average_yaw_rate * duration * cycle_time
        # With freq=1.0, one phase cycle = 1.0s, each sub-phase = 0.2s
        
        self.pulse_definitions = [
            # [phase_start, phase_end, peak_yaw_rate (rad/s), direction_sign]
            [0.0, 0.2, 5.0, 1.0],   # 60° CCW: ~5.0 rad/s avg over 0.2s → ~1.047 rad
            [0.2, 0.4, 3.3, -1.0],  # 40° CW:  ~3.3 rad/s avg over 0.2s → ~0.698 rad
            [0.4, 0.6, 2.1, 1.0],   # 25° CCW: ~2.1 rad/s avg over 0.2s → ~0.436 rad
            [0.6, 0.8, 1.3, -1.0],  # 15° CW:  ~1.3 rad/s avg over 0.2s → ~0.262 rad
            [0.8, 1.0, 0.4, 1.0],   # 5° CCW:  ~0.4 rad/s avg over 0.2s → ~0.087 rad
        ]
        
        # Radial extension parameters
        # Maximum radial extension factor during pulses
        self.radial_extension_amplitudes = [
            0.15,  # Phase 0.0-0.2: strong extension
            -0.10, # Phase 0.2-0.4: retraction (negative = inward)
            0.08,  # Phase 0.4-0.6: moderate extension
            -0.04, # Phase 0.6-0.8: slight retraction
            0.0,   # Phase 0.8-1.0: return to neutral
        ]

    def update_base_motion(self, phase, dt):
        """
        Update base pose with damped oscillatory yaw rotation.
        Linear velocities remain zero (in-place rotation).
        """
        # Determine current yaw rate based on phase and pulse profile
        yaw_rate = self._compute_yaw_rate(phase)
        
        # Zero linear velocity (in-place rotation)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        
        # Pure yaw rotation (zero roll and pitch rates)
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose in world frame
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def _compute_yaw_rate(self, phase):
        """
        Compute instantaneous yaw rate based on current phase.
        Uses smooth sinusoidal envelope within each pulse sub-phase.
        """
        for i, (p_start, p_end, peak_rate, sign) in enumerate(self.pulse_definitions):
            if p_start <= phase < p_end:
                # Normalized progress within this sub-phase [0, 1]
                local_progress = (phase - p_start) / (p_end - p_start)
                
                # Smooth sinusoidal envelope: starts at 0, peaks at 0.5, returns to 0
                envelope = np.sin(np.pi * local_progress)
                
                # Apply peak rate and direction
                yaw_rate = sign * peak_rate * envelope
                
                return yaw_rate
        
        # Default: zero yaw rate (should not reach here if phase in [0,1])
        return 0.0

    def _compute_radial_modulation(self, phase):
        """
        Compute radial extension/retraction factor based on phase.
        Returns a scalar multiplier for radial distance adjustment.
        """
        for i, (p_start, p_end, _, _) in enumerate(self.pulse_definitions):
            if p_start <= phase < p_end:
                # Normalized progress within sub-phase
                local_progress = (phase - p_start) / (p_end - p_start)
                
                # Smooth sinusoidal modulation within sub-phase
                # Amplitude corresponds to this sub-phase's radial extension
                amplitude = self.radial_extension_amplitudes[i]
                
                # Smooth transition using sine wave
                modulation = amplitude * np.sin(np.pi * local_progress)
                
                return modulation
        
        return 0.0

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with radial stance modulation.
        All feet remain in contact; radial distance varies to modulate inertia.
        """
        # Start with nominal foot position
        foot_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial modulation factor for current phase
        radial_factor = self._compute_radial_modulation(phase)
        
        # Calculate current radial direction (normalized)
        radial_2d = np.sqrt(foot_pos[0]**2 + foot_pos[1]**2)
        if radial_2d > 1e-6:
            radial_dir = np.array([foot_pos[0] / radial_2d, foot_pos[1] / radial_2d])
        else:
            # Fallback for feet at origin (shouldn't happen)
            radial_dir = np.array([1.0, 0.0])
        
        # Apply radial extension/retraction in horizontal plane
        foot_pos[0] += radial_factor * radial_dir[0]
        foot_pos[1] += radial_factor * radial_dir[1]
        
        # Z coordinate remains unchanged (feet stay on ground)
        # foot_pos[2] is unchanged
        
        return foot_pos