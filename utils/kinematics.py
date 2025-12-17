import numpy as np
import mujoco

class MultiLinkGradientDescentIK:
    """
    Multi-link Gradient Descent IK solver using stacked Jacobians.

    Args:
        model (mujoco.MjModel): MuJoCo model
        data (mujoco.MjData): MuJoCo data
        body_names (list[str]): list of bodies (e.g., ["FL_calf", "FR_calf", ...])
        step_size (float): IK gradient step size
        tol (float): tolerance for convergence (meters)
    """
    def __init__(self, model, data, body_names, step_size=0.01, tol=1e-3):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol

        # Body IDs
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]

        # Site IDs (foot tips)
        self.site_ids = []
        for name in body_names:
            # Guess foot site name
            site_name = name.replace("_calf", "_foot")
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                print(f"⚠️  Site {site_name} not found; using body {name} position")
                sid = None
            self.site_ids.append(sid)

        # Joint DOFs
        self.dof_ids = self._get_leg_dofs()
        print(f"[IK] Initialized: {len(self.dof_ids)} DOFs, {len(self.site_ids)} sites")

    def _get_leg_dofs(self):
        """Get joint indices for all legs"""
        dof_ids = []
        for leg in ["FL", "FR", "RL", "RR"]:
            for joint in ["hip", "thigh", "calf"]:
                jn = f"{leg}_{joint}_joint"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid >= 0:
                    dof_ids.append(self.model.jnt_dofadr[jid])
        return np.array(dof_ids, dtype=int)

    def get_foot_positions(self):
        """Get current foot positions in world frame"""
        pos = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            if sid is not None:
                pos.append(self.data.site_xpos[sid].copy())
            else:
                pos.append(self.data.xpos[bid].copy())
        return np.array(pos)

    def damped_least_squares(self, J, dx, damping=1e-3):
        """Damped least squares solution"""
        H = J.T @ J + (damping ** 2) * np.eye(J.shape[1])
        dq = np.linalg.solve(H, J.T @ dx)
        return dq

    def calculate(self, target_feet_pos, max_iter=500, damping=1e-3, debug=False):
        """Compute IK solution to reach target foot positions"""
        mujoco.mj_fwdPosition(self.model, self.data)
        dq = np.zeros(self.model.nv)

        for it in range(max_iter):
            current = self.get_foot_positions()
            err = (target_feet_pos - current).flatten()
            avg_err = np.linalg.norm(err) / len(self.site_ids)
            if avg_err < self.tol:
                break

            # Stacked Jacobian
            J_list = []
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            for sid in self.site_ids:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
                J_list.append(jacp[:, self.dof_ids].copy())
            J = np.vstack(J_list)

            dq_local = self.step_size * self.damped_least_squares(J, err, damping)
            dq[self.dof_ids] = dq_local
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1)
            mujoco.mj_fwdPosition(self.model, self.data)

        if debug:
            print(f"[IK] Converged in {it} iters, avg_err={avg_err*1000:.2f} mm")
        return avg_err
