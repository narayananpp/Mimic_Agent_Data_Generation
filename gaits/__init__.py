from gaits.walking import WalkingGaitController
from gaits.skating import SkatingGaitController
from gaits.static import StaticGaitController

# gaits/__init__.py
GAIT_REGISTRY = {
    "walking": WalkingGaitController,
    "skating": SkatingGaitController,
    "static": StaticGaitController

}

def get_gait_controller(name):
    return GAIT_REGISTRY[name]  # <-- return the class, do not call it

