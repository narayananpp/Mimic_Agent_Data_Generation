from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating yaw pulse skill: damped oscillatory yaw rotation settling to +45 degrees.
    
    Motion: Five pulses with decreasing amplitude (60°, -40°, +25°, -15°, +5°)
    All four legs remain in contact throughout, repositioning in body frame
    to generate/absorb angular momentum through radial extension/retraction.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        
        self.leg_names = leg_names
        self.freq = 1.0  # One full reverberating cycle per second
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time tracking
        self.t = 0.0
        
        # Base state (world frame)
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase boundaries
        self.phase_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Yaw pulse specifications: (target_delta_degrees, sign)
        # Tuned to accumulate net +45 degrees over full cycle
        self.pulse_specs = [
            (60.0, 1.0),   # Initial CCW pulse
            (40.0, -1.0),  # First CW reversal
            (25.0, 1.0),   # Second CCW correction
            (15.0, -1.0),  # Second CW correction
            (5.0, 1.0)     # Final damping
        ]
        
        # Radial extension amplitudes (damped progression)
        # Maximum extension during first pulse, decaying thereafter
        self.radial_amplitudes = [0.05, 0.03, 0.02, 0.01, 0.005]
        
        # Diagonal coordination groups
        self.diagonal_group_1 = []  # FL, RR
        self.diagonal_group_2 = []  # FR, RL
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.diagonal_group_1.append(leg)
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.diagonal_group_2.append(leg)

    def get_current_pulse_index(self, phase):
        """Determine which pulse phase we're in."""
        for i in range(len(self.phase_boundaries) - 1):
            if self.phase_boundaries[i] <= phase < self.phase_boundaries[i + 1]:
                return i
        return len(self.pulse_specs) - 1

    def get_local_phase(self, phase, pulse_idx):
        """Compute normalized phase within current pulse [0, 1]."""
        phase_start = self.phase_boundaries[pulse_idx]
        phase_end = self.phase_boundaries[pulse_idx + 1]
        return (phase - phase_start) / (phase_end - phase_start)

    def compute_yaw_rate(self, phase):
        """
        Compute smooth yaw rate with ramp-up/down at pulse boundaries.
        Uses smooth cosine profile within each pulse.
        """
        pulse_idx = self.get_current_pulse_index(phase)
        local_phase = self.get_local_phase(phase, pulse_idx)
        
        target_delta_deg, sign = self.pulse_specs[pulse_idx]
        target_delta_rad = np.deg2rad(target_delta_deg)
        
        # Duration of this pulse
        duration = self.phase_boundaries[pulse_idx + 1] - self.phase_boundaries[pulse_idx]
        phase_duration_sec = duration / self.freq
        
        # Smooth cosine envelope: ramps up, sustains, ramps down
        # Peak yaw rate needed to accumulate target_delta_rad over phase_duration_sec
        # Using smooth raised cosine window
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * local_phase))
        
        # Average yaw rate needed (integral of window over [0,1] is 0.5)
        # So peak rate should be 2 * (target_delta_rad / phase_duration_sec)
        peak_yaw_rate = 2.0 * target_delta_rad / phase_duration_sec
        
        yaw_rate = sign * peak_yaw_rate * window
        
        return yaw_rate

    def update_base_motion(self, phase, dt):
        """
        Update base using computed yaw rate. No linear velocity (in-place rotation).
        """
        yaw_rate = self.compute_yaw_rate(phase)
        
        # Zero linear velocity (in-place motion)
        self.vel_world = np.array([0.0, 0.0, 0.0])
        
        # Pure yaw rotation
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )

    def compute_radial_offset(self, leg_name, phase):
        """
        Compute radial extension/retraction in body frame.
        Extension during CCW pulses, retraction during CW pulses.
        Diagonal groups coordinate symmetrically.
        """
        pulse_idx = self.get_current_pulse_index(phase)
        local_phase = self.get_local_phase(phase, pulse_idx)
        
        _, sign = self.pulse_specs[pulse_idx]
        amplitude = self.radial_amplitudes[pulse_idx]
        
        # Smooth radial motion using sine envelope
        radial_factor = np.sin(np.pi * local_phase)
        
        # Diagonal group 1 (FL, RR) extends during CCW (sign=1), retracts during CW (sign=-1)
        # Diagonal group 2 (FR, RL) follows same pattern (all legs extend/retract together for balance)
        radial_offset = sign * amplitude * radial_factor
        
        return radial_offset

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position in body frame with radial extension/retraction.
        All feet remain in contact; repositioning creates angular momentum.
        """
        base_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Compute radial direction from body center to foot
        radial_dir = base_pos[:2] / (np.linalg.norm(base_pos[:2]) + 1e-6)
        
        # Compute radial offset magnitude
        radial_offset_mag = self.compute_radial_offset(leg_name, phase)
        
        # Apply radial offset in x-y plane
        foot = base_pos.copy()
        foot[0] += radial_dir[0] * radial_offset_mag
        foot[1] += radial_dir[1] * radial_offset_mag
        
        # Z remains constant (no vertical motion, all feet stay grounded)
        
        return foot