import numpy as np
files = [
    "1773947469650_skill7_spot_frames.npz",
    "1773947469650_skill7_barkour_vb_frames.npz",
    "1773947469650_skill7_anymal_c_frames.npz",
]

for f in files:
    data = np.load(f)
    frames = data["frames"]
