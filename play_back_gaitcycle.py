import mujoco
import glfw
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

def main():
    # -------------------------------
    # Load recorded gait reference
    # -------------------------------
    gait_file = Path("data/skating_gait_reference_v0.6_f2.40.pkl")
    if not gait_file.exists():
        raise FileNotFoundError(f"❌ Gait file not found: {gait_file.resolve()}")

    with open(gait_file, "rb") as f:
        gait_data = pickle.load(f)

    times = np.array(gait_data["time"])          # shape (T,)
    qpos_log = np.array(gait_data["qpos"])       # (T, nq)
    foot_body = np.array(gait_data["foot_body"]) # (T, 4, 3)
    print(f"✅ Loaded gait data: {len(times)} frames, duration={times[-1]:.3f}s")

    # -------------------------------
    # Normalize time for phase-based snapshots
    # -------------------------------
    normalized_time = (times - times[0]) / (times[-1] - times[0])
    phases = [0.0, 0.5, 1.0]
    phase_indices = [np.argmin(np.abs(normalized_time - p)) for p in phases]
    print(f"📸 Snapshot indices (for 0, 0.5, 1 phase): {phase_indices}")

    # -------------------------------
    # Load MuJoCo model
    # -------------------------------
    xml_path = "mujoco_menagerie/unitree_go2/scene_mjx.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # -------------------------------
    # GLFW Setup
    # -------------------------------
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")
    window = glfw.create_window(1280, 900, "Go2 Gait Playback", None, None)
    glfw.make_context_current(window)

    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.3]
    cam.distance = 2.5
    cam.azimuth, cam.elevation = 90, -20
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    # -------------------------------
    # Playback parameters
    # -------------------------------
    fps = 200
    duration = 2.0
    num_frames = int(fps * duration)
    dt = 1.0 / fps
    print(f"▶️ Capturing {duration}s ({num_frames} frames @ {fps}Hz)")

    # -------------------------------
    # Capture frames for video
    # -------------------------------
    frames = []
    snapshot_images = {}

    for i in range(num_frames):
        idx = i % len(qpos_log)
        data.qpos[:] = qpos_log[idx]
        mujoco.mj_fwdPosition(model, data)

        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 1280, 900), scene, context)

        width, height = glfw.get_framebuffer_size(window)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, width, height), context)
        rgb = np.flipud(rgb)

        frames.append(rgb)

        # Capture exact normalized phase snapshots
        if idx in phase_indices and idx not in snapshot_images:
            phase_val = normalized_time[idx]
            snapshot_images[idx] = (phase_val, rgb.copy())

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

    # -------------------------------
    # Save GIF
    # -------------------------------
    imageio.mimsave("media/gait_playback.gif", frames[::2], fps=fps//2)
    print("🎞️  Saved: gait_playback.gif")

    # -------------------------------
    # Save phase snapshots
    # -------------------------------
    for idx, (phase_val, img) in snapshot_images.items():
        fname = f"media/snapshot_phase_{phase_val:.1f}.png"
        imageio.imwrite(fname, img)
        print(f"📸 Saved snapshot: {fname}")

    # -------------------------------
    # Plot z and x trajectories (all legs)
    # -------------------------------
    legs = ["FL", "FR", "RL", "RR"]
    colors = ["r", "g", "b", "m"]

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(times, foot_body[:, i, 2], label=f"{legs[i]} (z)", color=colors[i])
    plt.title("Foot height (z) trajectories")
    plt.xlabel("Time [s]")
    plt.ylabel("Height [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/foot_z_trajectories.png", dpi=200)
    print("📈 Saved: foot_z_trajectories.png")

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(times, foot_body[:, i, 0], label=f"{legs[i]} (x)", color=colors[i])
    plt.title("Foot forward (x) trajectories")
    plt.xlabel("Time [s]")
    plt.ylabel("Forward displacement [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/foot_x_trajectories.png", dpi=200)
    plt.show()
    print("📈 Saved: foot_x_trajectories.png")

if __name__ == "__main__":
    main()
