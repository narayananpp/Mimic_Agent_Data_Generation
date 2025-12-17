from abc import ABC, abstractmethod

class BaseGaitController(ABC):
    """
    Abstract base class for all gait controllers.
    Ensures a consistent interface for all skills.
    """

    def __init__(self, base_init_feet_pos, freq=1.0):
        self.freq = freq
        self.base_init_feet_pos = base_init_feet_pos.copy()
        self.base_feet_pos = base_init_feet_pos.copy()

    @abstractmethod
    def set_base_init_feet_pos(self, vx=0.0, yaw=0.0, dt=0.002, yaw_rate=0.0):
        """Shift reference foot positions forward as the body moves."""
        pass

    @abstractmethod
    def foot_target(self, leg_name, t, **kwargs):
        """Return target foot position for a given leg at time t."""
        pass

    def com_height(self, t):
        """Optional: Return center-of-mass height at time t."""
        return None

    def base_orientation(self, t):
        """Optional: Return base orientation (quaternion) at time t."""
        return None
