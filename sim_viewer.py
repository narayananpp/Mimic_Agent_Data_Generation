import mujoco
import mujoco.viewer

# --- Path to your MuJoCo XML file ---
# Example: Unitree Go2 from Menagerie
xml_path = "mujoco_menagerie/unitree_go2w/go2.xml"

# --- Load the model ---
model = mujoco.MjModel.from_xml_path(xml_path)

# --- Create simulation data ---
data = mujoco.MjData(model)

# --- Launch viewer ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer running... Close the window to exit.")
    
    # Run until the viewer is closed
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
