from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np

class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating yaw pulse skill: damped oscillatory yaw rotation in place.
    
    Five alternating yaw pulses with decreasing amplitude:
    - Pulse 1 (CCW): 60 degrees
    - Pulse 2 (CW):  40 degrees
    - Pulse 3 (CCW): 25 degrees
    - Pulse 4 (CW):  15 degrees
    - Pulse 5 (CCW): 5 degrees
    
    Net rotation: ~45 degrees CCW from start.
    All feet remain in contact throughout, with radial extension/retraction
    to modulate rotational inertia.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 1.0  # One full cycle through all five pulses
        
        # Base foot positions (BODY frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase boundaries for the five pulses
        self.phase_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Target angular displacements per pulse (in degrees)
        self.pulse_angles = [60.0, -40.0, 25.0, -15.0, 5.0]
        
        # Convert to yaw rates (rad/s) for each phase
        # Each phase lasts 0.2 cycles, duration = 0.2 / freq seconds
        phase_duration = 0.2 / self.freq
        self.yaw_rates = [
            np.deg2rad(angle) / phase_duration for angle in self.pulse_angles
        ]
        
        # Radial extension parameters
        # Extension amplitudes relative to first pulse (qualitative tuning)
        self.extension_amplitudes = [1.0, -0.6, 0.6, -0.4, 0.0]
        self.max_radial_extension = 0.05  # meters, tuned for stability
        
    def update_base_motion(self, phase, dt):
        """
        Update base orientation using phase-dependent yaw rate.
        Linear velocity remains zero (in-place rotation).
        """
        # Determine which pulse phase we're in
        pulse_index = self._get_pulse_index(phase)
        
        # Get yaw rate for current pulse with smooth transitions
        yaw_rate = self._get_smooth_yaw_rate(phase, pulse_index)
        
        # No linear motion, pure yaw rotation
        self.vel_world = np.array([0.0, 0.0, 0.0])
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Integrate pose
        self.root_pos, self.root_quat = integrate_pose_world_frame(
            self.root_pos,
            self.root_quat,
            self.vel_world,
            self.omega_world,
            dt
        )
    
    def _get_pulse_index(self, phase):
        """Determine which of the five pulses we're currently in."""
        for i in range(len(self.phase_boundaries) - 1):
            if self.phase_boundaries[i] <= phase < self.phase_boundaries[i + 1]:
                return i
        return 4  # Last pulse
    
    def _get_smooth_yaw_rate(self, phase, pulse_index):
        """
        Get yaw rate with smooth ramping at phase boundaries to avoid
        discontinuous angular accelerations.
        """
        base_yaw_rate = self.yaw_rates[pulse_index]
        
        # Smooth ramping parameters
        ramp_fraction = 0.1  # Use 10% of phase duration for ramping
        phase_start = self.phase_boundaries[pulse_index]
        phase_end = self.phase_boundaries[pulse_index + 1]
        phase_duration = phase_end - phase_start
        ramp_duration = phase_duration * ramp_fraction
        
        # Compute local phase within current pulse
        local_phase = (phase - phase_start) / phase_duration
        
        # Apply smooth ramping at start and end of pulse
        if local_phase < ramp_fraction:
            # Ramp up from previous pulse
            ramp_progress = local_phase / ramp_fraction
            smooth_factor = 0.5 * (1.0 - np.cos(np.pi * ramp_progress))
            
            # Blend with previous pulse yaw rate
            if pulse_index > 0:
                prev_yaw_rate = self.yaw_rates[pulse_index - 1]
            else:
                prev_yaw_rate = 0.0
            
            return prev_yaw_rate * (1.0 - smooth_factor) + base_yaw_rate * smooth_factor
        
        elif local_phase > (1.0 - ramp_fraction):
            # Ramp down to next pulse
            ramp_progress = (local_phase - (1.0 - ramp_fraction)) / ramp_fraction
            smooth_factor = 0.5 * (1.0 - np.cos(np.pi * ramp_progress))
            
            # Blend with next pulse yaw rate
            if pulse_index < 4:
                next_yaw_rate = self.yaw_rates[pulse_index + 1]
            else:
                next_yaw_rate = 0.0
            
            return base_yaw_rate * (1.0 - smooth_factor) + next_yaw_rate * smooth_factor
        
        else:
            # Middle of pulse, use base yaw rate
            return base_yaw_rate
    
    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position with radial extension/retraction to modulate
        rotational inertia while maintaining ground contact.
        """
        # Start from base foot position
        foot = self.base_feet_pos_body[leg_name].copy()
        
        # Determine pulse index
        pulse_index = self._get_pulse_index(phase)
        
        # Get radial extension amplitude for current pulse
        extension_amplitude = self.extension_amplitudes[pulse_index]
        
        # Compute radial direction from body center to foot (in XY plane)
        radial_xy = np.array([foot[0], foot[1]])
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance > 1e-6:
            radial_direction = radial_xy / radial_distance
        else:
            # Fallback if foot is at body center
            if leg_name.startswith('FL'):
                radial_direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
            elif leg_name.startswith('FR'):
                radial_direction = np.array([1.0, -1.0]) / np.sqrt(2.0)
            elif leg_name.startswith('RL'):
                radial_direction = np.array([-1.0, 1.0]) / np.sqrt(2.0)
            else:  # RR
                radial_direction = np.array([-1.0, -1.0]) / np.sqrt(2.0)
        
        # Apply smooth radial extension/retraction within current pulse
        phase_start = self.phase_boundaries[pulse_index]
        phase_end = self.phase_boundaries[pulse_index + 1]
        local_phase = (phase - phase_start) / (phase_end - phase_start)
        
        # Use smooth sinusoidal profile for extension within pulse
        extension_profile = np.sin(np.pi * local_phase)
        radial_offset = extension_amplitude * self.max_radial_extension * extension_profile
        
        # Apply radial offset to foot position
        foot[0] += radial_direction[0] * radial_offset
        foot[1] += radial_direction[1] * radial_offset
        
        # Z remains at ground contact (no vertical motion)
        # foot[2] unchanged
        
        return foot