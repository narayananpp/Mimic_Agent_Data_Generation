from utils.math_utils import *
from gaits.base import BaseMotionGenerator

class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating yaw pulse skill: executes damped oscillatory yaw rotations
    with alternating direction and decreasing amplitude, settling to 45° CCW.
    
    Pulse sequence:
      - Phase [0.0, 0.2]: 60° CCW
      - Phase [0.2, 0.4]: 40° CW
      - Phase [0.4, 0.6]: 25° CCW
      - Phase [0.6, 0.8]: 15° CW
      - Phase [0.8, 1.0]: 5° CCW
    
    All feet remain in stance throughout. Legs extend/retract radially
    to modulate rotational inertia during pulses.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Full reverb cycle takes 2 seconds at default
        
        # Base foot positions (body frame)
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Phase boundaries for five pulses
        self.phase_boundaries = [
            (0.0, 0.2),   # Pulse 1: 60° CCW
            (0.2, 0.4),   # Pulse 2: 40° CW
            (0.4, 0.6),   # Pulse 3: 25° CCW
            (0.6, 0.8),   # Pulse 4: 15° CW
            (0.8, 1.0),   # Pulse 5: 5° CCW
        ]
        
        # Target rotations per pulse (in radians)
        self.pulse_rotations = [
            np.deg2rad(60),   # CCW
            np.deg2rad(-40),  # CW (negative)
            np.deg2rad(25),   # CCW
            np.deg2rad(-15),  # CW
            np.deg2rad(5),    # CCW
        ]
        
        # Leg extension magnitudes per pulse (radial offset from neutral)
        # Scales with pulse amplitude to create momentum modulation effect
        self.extension_magnitudes = [
            0.08,   # Maximum extension for largest pulse
            0.05,   # Retraction during reversal
            0.04,   # Moderate extension
            0.02,   # Small adjustment
            0.01,   # Minimal settling adjustment
        ]
        
        # Base state
        self.t = 0.0
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Velocity commands (zero linear velocity throughout)
        self.vel_world = np.zeros(3)
        self.omega_world = np.zeros(3)

    def _get_pulse_index(self, phase):
        """Determine which pulse phase we're in."""
        for i, (p_start, p_end) in enumerate(self.phase_boundaries):
            if p_start <= phase < p_end:
                return i
        return 4  # Last pulse (phase == 1.0 edge case)

    def _smooth_transition(self, t, t_start, t_end):
        """Sigmoid-like smooth transition from 0 to 1 over [t_start, t_end]."""
        if t <= t_start:
            return 0.0
        if t >= t_end:
            return 1.0
        # Normalized progress in [0, 1]
        s = (t - t_start) / (t_end - t_start)
        # Smoothstep function
        return s * s * (3.0 - 2.0 * s)

    def update_base_motion(self, phase, dt):
        """
        Compute yaw rate based on current pulse phase.
        Each pulse phase has a target rotation; angular velocity is computed
        to integrate to that rotation over the phase duration.
        """
        pulse_idx = self._get_pulse_index(phase)
        p_start, p_end = self.phase_boundaries[pulse_idx]
        phase_duration = p_end - p_start
        
        # Target rotation for this pulse
        target_rotation = self.pulse_rotations[pulse_idx]
        
        # Time duration for this pulse (in seconds)
        time_duration = phase_duration / self.freq
        
        # Average angular velocity needed to achieve target rotation
        # Use sinusoidal profile within pulse for smooth acceleration/deceleration
        local_phase = (phase - p_start) / phase_duration  # in [0, 1]
        
        # Sinusoidal velocity profile: peaks at mid-pulse, zero at boundaries
        # Integral of sin over [0, π] is 2, so scale amplitude accordingly
        velocity_amplitude = target_rotation * np.pi / (2.0 * time_duration)
        yaw_rate = velocity_amplitude * np.sin(np.pi * local_phase)
        
        # Set angular velocity (zero roll and pitch)
        self.omega_world = np.array([0.0, 0.0, yaw_rate])
        
        # Zero linear velocity (in-place rotation)
        self.vel_world = np.zeros(3)
        
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
        Compute foot position in body frame with radial extension/retraction.
        
        During CCW pulses: legs extend radially to increase moment of inertia.
        During CW pulses: legs retract to reduce inertia and enable reversal.
        """
        pulse_idx = self._get_pulse_index(phase)
        p_start, p_end = self.phase_boundaries[pulse_idx]
        
        # Get base foot position
        base_foot = self.base_feet_pos_body[leg_name].copy()
        
        # Current extension magnitude for this pulse
        extension = self.extension_magnitudes[pulse_idx]
        
        # Smooth transition within pulse (ramp up at start, ramp down at end)
        local_phase = (phase - p_start) / (p_end - p_start)
        
        # Use sinusoidal modulation for smooth extension profile
        extension_factor = np.sin(np.pi * local_phase)
        
        # Compute radial direction in x-y plane (body frame)
        radial_xy = np.array([base_foot[0], base_foot[1]])
        radial_distance = np.linalg.norm(radial_xy)
        
        if radial_distance > 1e-6:
            radial_direction = radial_xy / radial_distance
        else:
            # Fallback for legs near center (shouldn't happen with typical stance)
            radial_direction = np.array([1.0, 0.0])
        
        # Apply radial offset
        # For CCW pulses (positive rotation): extend outward
        # For CW pulses (negative rotation): retract inward (negative extension)
        if self.pulse_rotations[pulse_idx] > 0:
            offset_xy = extension * extension_factor * radial_direction
        else:
            offset_xy = -extension * extension_factor * radial_direction
        
        # Construct foot position
        foot = base_foot.copy()
        foot[0] += offset_xy[0]
        foot[1] += offset_xy[1]
        
        # Maintain constant z (no vertical motion, all feet in contact)
        return foot