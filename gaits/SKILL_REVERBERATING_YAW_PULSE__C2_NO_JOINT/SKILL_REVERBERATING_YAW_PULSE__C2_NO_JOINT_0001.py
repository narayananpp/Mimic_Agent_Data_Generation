from utils.math_utils import *
from gaits.base import BaseMotionGenerator
import numpy as np


class SKILL_REVERBERATING_YAW_PULSE_MotionGenerator(BaseMotionGenerator):
    """
    Reverberating Yaw Pulse: In-place rotational maneuver with damped oscillations.
    
    The robot executes a series of yaw pulses with decreasing amplitude:
    - Phase [0.0, 0.2]: 60° CCW pulse
    - Phase [0.2, 0.4]: 40° CW reversal
    - Phase [0.4, 0.6]: 25° CCW reversal
    - Phase [0.6, 0.8]: 15° CW correction
    - Phase [0.8, 1.0]: 5° CCW settle
    
    Net rotation: ~45° counterclockwise
    All four feet remain in contact throughout, with coordinated radial
    extensions/retractions to generate yaw moments.
    """

    def __init__(self, initial_foot_positions_body, leg_names):
        self.leg_names = leg_names
        self.freq = 0.5  # Slow cycle to allow observable oscillations
        
        # Store nominal foot positions in body frame
        self.base_feet_pos_body = {k: v.copy() for k, v in initial_foot_positions_body.items()}
        
        # Time tracking
        self.t = 0.0
        
        # Base state
        self.root_pos = np.zeros(3)
        self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Phase boundaries for sub-phases
        self.phase_boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Target rotation angles per phase (in degrees)
        # CCW is positive, CW is negative
        self.target_rotations = [60.0, -40.0, 25.0, -15.0, 5.0]
        
        # Convert to radians
        self.target_rotations_rad = [np.deg2rad(angle) for angle in self.target_rotations]
        
        # Compute angular velocities needed per phase
        # Each phase has duration = (phase_end - phase_start) / freq
        self.angular_velocities = []
        for i in range(len(self.target_rotations_rad)):
            phase_duration = (self.phase_boundaries[i+1] - self.phase_boundaries[i]) / self.freq
            yaw_rate = self.target_rotations_rad[i] / phase_duration
            self.angular_velocities.append(yaw_rate)
        
        # Radial extension parameters for leg motion
        # Extension magnitude decreases with damping across phases
        self.extension_magnitudes = [0.12, 0.09, 0.06, 0.04, 0.02]  # meters
        
        # Diagonal leg grouping for coordinated motion
        # Group 1 (FL, RR): extend during CCW, retract during CW
        # Group 2 (FR, RL): retract during CCW, extend during CW
        self.group_1 = []
        self.group_2 = []
        for leg in leg_names:
            if leg.startswith('FL') or leg.startswith('RR'):
                self.group_1.append(leg)
            elif leg.startswith('FR') or leg.startswith('RL'):
                self.group_2.append(leg)

    def update_base_motion(self, phase, dt):
        """
        Apply yaw angular velocity commands based on current phase.
        Linear velocities kept at zero for in-place rotation.
        """
        # Determine which sub-phase we're in
        sub_phase_idx = self._get_sub_phase_index(phase)
        
        # Get local phase within current sub-phase for smooth transitions
        local_phase = self._get_local_phase(phase, sub_phase_idx)
        
        # Base yaw rate for this sub-phase
        base_yaw_rate = self.angular_velocities[sub_phase_idx]
        
        # Apply smooth ramping at phase boundaries using sinusoidal envelope
        # This prevents jerky reversals
        ramp_factor = self._get_smooth_ramp(local_phase)
        yaw_rate = base_yaw_rate * ramp_factor
        
        # Set velocity commands (zero linear velocity for in-place motion)
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

    def compute_foot_position_body_frame(self, leg_name, phase):
        """
        Compute foot position with radial extension/retraction to assist yaw moments.
        All feet remain in contact (z position maintains ground contact).
        """
        # Start with nominal foot position
        foot_pos = self.base_feet_pos_body[leg_name].copy()
        
        # Determine which sub-phase we're in
        sub_phase_idx = self._get_sub_phase_index(phase)
        local_phase = self._get_local_phase(phase, sub_phase_idx)
        
        # Get extension magnitude for this phase (damped over time)
        max_extension = self.extension_magnitudes[sub_phase_idx]
        
        # Determine if this is a CCW or CW phase
        is_ccw_phase = self.target_rotations[sub_phase_idx] > 0
        
        # Determine leg group membership
        is_group_1 = leg_name in self.group_1  # FL or RR
        
        # Extension logic:
        # - During CCW phases: Group 1 extends, Group 2 retracts
        # - During CW phases: Group 1 retracts, Group 2 extends
        if is_ccw_phase:
            extension_sign = 1.0 if is_group_1 else -1.0
        else:
            extension_sign = -1.0 if is_group_1 else 1.0
        
        # Smooth extension profile using sinusoidal transition
        extension_profile = np.sin(np.pi * local_phase)
        extension_amount = extension_sign * max_extension * extension_profile
        
        # Apply radial extension/retraction in x-y plane
        # Compute radial direction from body center to nominal foot position
        radial_direction = np.array([foot_pos[0], foot_pos[1]])
        radial_distance = np.linalg.norm(radial_direction)
        
        if radial_distance > 1e-6:
            radial_unit = radial_direction / radial_distance
            foot_pos[0] += radial_unit[0] * extension_amount
            foot_pos[1] += radial_unit[1] * extension_amount
        
        # Maintain ground contact - z position stays at nominal (no vertical motion)
        # foot_pos[2] remains unchanged from base position
        
        return foot_pos

    def _get_sub_phase_index(self, phase):
        """
        Determine which sub-phase index corresponds to current phase value.
        """
        for i in range(len(self.phase_boundaries) - 1):
            if self.phase_boundaries[i] <= phase < self.phase_boundaries[i+1]:
                return i
        # Handle edge case where phase == 1.0
        return len(self.phase_boundaries) - 2

    def _get_local_phase(self, phase, sub_phase_idx):
        """
        Compute normalized local phase [0,1] within the current sub-phase.
        """
        phase_start = self.phase_boundaries[sub_phase_idx]
        phase_end = self.phase_boundaries[sub_phase_idx + 1]
        phase_duration = phase_end - phase_start
        
        if phase_duration < 1e-9:
            return 0.0
        
        local_phase = (phase - phase_start) / phase_duration
        return np.clip(local_phase, 0.0, 1.0)

    def _get_smooth_ramp(self, local_phase):
        """
        Apply smooth sinusoidal ramping at phase boundaries.
        Ramps up at start and down at end of each sub-phase to avoid jerky transitions.
        """
        ramp_region = 0.15  # 15% of phase duration for ramping
        
        if local_phase < ramp_region:
            # Ramp up from 0 to 1
            ramp_progress = local_phase / ramp_region
            return 0.5 * (1.0 - np.cos(np.pi * ramp_progress))
        elif local_phase > (1.0 - ramp_region):
            # Ramp down from 1 to 0
            ramp_progress = (local_phase - (1.0 - ramp_region)) / ramp_region
            return 0.5 * (1.0 + np.cos(np.pi * ramp_progress))
        else:
            # Full magnitude in middle of phase
            return 1.0