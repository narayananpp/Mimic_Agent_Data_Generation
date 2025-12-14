# Go2 Kinematic Gait & Skating Motion Generator (MuJoCo + MimicKit)

#### This repository provides a **kinematic Walking and Skating gait motion generator**  for the **Unitree Go2 and Go2W** robot using **MuJoCo**, **inverse kinematics**, and **procedural gait controllers**. The generated animated motions are exported in **MimicKit-compatible format** (`.pkl`) for downstream reinforcement learning using DeepMimic and other methods for getting a policy that generate similar but dynamically feasible motion.
---

## Features

- Walking, skating, and static motion modes  
- Multiple skating styles (front/back/side/diagonal/two-leg)  
- Real-time MuJoCo visualization  
- Optional live foot trajectory plotting  
- YAML-based configuration with CLI overrides  
- Exports loopable MimicKit motion files  
- Pure kinematic IK (no physics simulation)

---

### 1. Create environment
```bash
conda create -n mujoco python=3.10
conda activate mujoco 

conda install -c conda-forge mujoco glfw pyyaml numpy matplotlib
pip install pickle5
```
### 2. Running the scipt

Running the Script

From the scripts/ directory:

```
cd scripts
```

Run with default configuration:

```
python sim_go2_mimickit.py
```

### 3. Supported Modes
#### Walking

Standard quadruped walking gait.

```
python sim_go2_mimickit.py --mode walking
```

#### Skating

Passive-wheel skating with selectable styles.
```
python sim_go2_mimickit.py --mode skating --skating_style back_alt
```
#### Static

Feet remain fixed (debug/visualization).
```
python sim_go2_mimickit.py --mode static
```
### 4. Additional args 
Live Plotting

Enable live plotting of foot trajectories:
```
--plot true
```

Plots:

Foot Z-height

Foot X-position

Recording MimicKit Motions

#### Enable recording:
```
--record true --record_cycles 4
```

#### Saved output:
```
data/go2_<skating_style>_<mode>_gait_mimickit.pkl
```
### 4. Output Format (MimicKit)

Each frame contains:
```
[root_position (3),
 root_expmap (3),
 joint_angles (...)]
```

Saved as:
```
{
  "fps": 200,
  "loop_mode": 1,
  "frames": np.ndarray [T, D]
}
```
