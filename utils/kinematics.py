"""
utils/kinematics.py

Multi-link gradient-descent IK solver.

Changes from original
---------------------
- No longer hardcodes FL/FR/RL/RR joint naming.
- Accepts explicit `foot_sites` and `joint_names` lists from RobotConfig,
  so it works with any robot (ANYmal, Go2W, Spot, etc.).
- Falls back to the old heuristic behaviour when those lists are omitted,
  for backward compatibility.
"""

import numpy as np
import mujoco


class MultiLinkGradientDescentIK:
    """
    Multi-link Gradient Descent IK solver using stacked Jacobians.

    Parameters
    ----------
    model       : mujoco.MjModel
    data        : mujoco.MjData
    body_names  : list[str]  — IK target bodies (XML names, e.g. calf bodies)
    foot_sites  : list[str] | None
        Explicit foot-tip site names in the same order as body_names.
        If None, the old heuristic (replace "_calf" → "_foot") is used.
    joint_names : list[str] | None
        All actuated joint names for the legs.
        If None, the old heuristic (FL/FR/RL/RR × hip/thigh/calf) is used.
    step_size   : float
    tol         : float
    """

    def __init__(
        self,
        model,
        data,
        body_names,
        foot_sites=None,
        joint_names=None,
        step_size=0.01,
        tol=1e-3,
    ):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol

        # ── Body IDs ────────────────────────────────────────────────────────
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in body_names
        ]

        # ── Site IDs (foot tips) ─────────────────────────────────────────────
        if foot_sites is not None:
            # Explicit list from robot config
            self.site_ids = []
            for site_name in foot_sites:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if sid < 0:
                    print(f"⚠️  Site '{site_name}' not found; falling back to body position")
                    sid = None
                self.site_ids.append(sid)
        else:
            # Legacy heuristic: replace body suffix to guess site name
            self.site_ids = []
            for name in body_names:
                site_name = name.replace("_calf", "_foot").replace("_shank", "_foot")
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if sid < 0:
                    print(f"⚠️  Site '{site_name}' not found; falling back to body position")
                    sid = None
                self.site_ids.append(sid)

        # ── DOF indices ──────────────────────────────────────────────────────
        if joint_names is not None:
            self.dof_ids = self._dofs_from_names(joint_names)
        else:
            self.dof_ids = self._dofs_heuristic()

        print(f"[IK] Robot bodies : {body_names}")
        print(f"[IK] DOFs found   : {len(self.dof_ids)}")
        print(f"[IK] Sites found  : {sum(s is not None for s in self.site_ids)}/{len(self.site_ids)}")

    # ── DOF helpers ─────────────────────────────────────────────────────────

    def _dofs_from_names(self, joint_names):
        """Build DOF index array from an explicit list of joint names."""
        dof_ids = []
        for jn in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                dof_ids.append(self.model.jnt_dofadr[jid])
            else:
                print(f"⚠️  Joint '{jn}' not found in model — skipping")
        return np.array(dof_ids, dtype=int)

    def _dofs_heuristic(self):
        """
        Legacy fallback: assumes FL/FR/RL/RR × hip/thigh/calf naming.
        Works for Unitree Go2W.  Will not work for ANYmal.
        """
        dof_ids = []
        for leg in ["FL", "FR", "RL", "RR"]:
            for joint in ["hip", "thigh", "calf"]:
                jn = f"{leg}_{joint}_joint"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    dof_ids.append(self.model.jnt_dofadr[jid])
        return np.array(dof_ids, dtype=int)

    # ── Forward kinematics ───────────────────────────────────────────────────

    def get_foot_positions(self):
        """Return current foot positions in world frame (N × 3)."""
        pos = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            if sid is not None:
                pos.append(self.data.site_xpos[sid].copy())
            else:
                pos.append(self.data.xpos[bid].copy())
        return np.array(pos)

    # ── IK solver ────────────────────────────────────────────────────────────

    def damped_least_squares(self, J, dx, damping=1e-3):
        H = J.T @ J + (damping ** 2) * np.eye(J.shape[1])
        return np.linalg.solve(H, J.T @ dx)

    def calculate(self, target_feet_pos, max_iter=500, damping=1e-3, debug=False):
        """
        Solve IK to place feet at target_feet_pos (N × 3, world frame).
        Modifies self.data.qpos in-place.
        Returns average position error (metres).
        """
        mujoco.mj_fwdPosition(self.model, self.data)
        dq = np.zeros(self.model.nv)
        avg_err = float("inf")

        for it in range(max_iter):
            current = self.get_foot_positions()
            err = (target_feet_pos - current).flatten()
            avg_err = np.linalg.norm(err) / max(len(self.site_ids), 1)
            if avg_err < self.tol:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            J_list = []
            for sid, bid in zip(self.site_ids, self.body_ids):
                if sid is not None:
                    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
                else:
                    mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bid)
                J_list.append(jacp[:, self.dof_ids].copy())

            J = np.vstack(J_list)
            dq_local = self.step_size * self.damped_least_squares(J, err, damping)
            dq[self.dof_ids] = dq_local
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1)
            mujoco.mj_fwdPosition(self.model, self.data)

        if debug:
            print(f"[IK] it={it:4d}  avg_err={avg_err * 1000:.2f} mm")

        return avg_err