import mujoco
import numpy as np
from gym_lowcostrobot.utils import rot_axis, wrap_neg_pi_pi

# ------------------------------- ANALYTICAL IK ------------------------------- #

def solve_2D_IK(x,y,l1,l2,elbow='down'):
    """Solve 2D inverse kinematics problem for a 2-link planar robot arm.

    The first joint is located at the origin O. The second joint P is at distance l1 from O. The
    end-effector E = (x,y) is at distance l2 from P. 
    alpha is the angle from [Oy) to [OP).
    beta is the angle from P + [OP) to [PE).
    
    There are two solutions for (alpha, beta): 
    "elbow down" (-pi <= beta <= 0) and "elbow up" (0 <= beta <= pi).

    Arguments:
    - x,y (float): desired position of the end-effector in the plane.
    - l1,l2 (float): lengths of the two links.
    - elbow (str): 'down', 'up' or 'both' (default='down').

    Returns:
    - (dict): angles of the two joints in radians of the selected solutions.
    - unreachable (bool): if the target is unreachable.

    Note: one must have (l1 - l2)² ≤ x² + y² ≤ (l1 + l2)².
    """
    theta = np.arctan2(y, x)
    cos_beta = (x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2)
    beta_up = np.arccos(np.clip(cos_beta, -1, 1))
    phi = np.arctan2(l2 * np.sin(beta_up), l1 + l2 * cos_beta)

    sol = {}
    if elbow == 'down' or elbow == 'both':
        sol['down'] = (wrap_neg_pi_pi(theta + phi - np.pi/2), -beta_up)
    if elbow == 'up' or elbow == 'both':
        sol['up'] = (wrap_neg_pi_pi(theta - phi - np.pi/2), beta_up)

    return sol, not (-1 <= cos_beta <= 1)


class InverseKinematicsZYYYX:
    """Inverse kinematics solver for a "z,y,y,y,x" 5 degrees of freedom robotic arm (e.g. Koch v1.1
    or SO-100 or SO-101). The yaw angle of the end-effector (EE) in world frame cannot be freely
    chosen, as it is totally determined by the EE's position and roll angle. The parameters that
    can be set freely are the EE's (x,y,z) coordinates and (roll, pitch) angles (in world frame).

    The kinematic tree is made of 8 frames, with 5 degrees of freedom:
     . world (w) 
    <- base_link (b): fixed in w.
    <- link_1 (1): rotates around z in b.
    <- link_2 (2): rotates around y in 1.
    <- link_3 (3): rotates around y in 2.
    <- link_4 (4): rotates around y in 3.
    <- link_5 (5): rotates around x in 4.
    <- end_effector_site (e): fixed in 5.
    """

    def __init__(self, model: mujoco.MjModel):
        # Notation:
        #  translation (t), rotation (R), coordinate (x,y,z), distance (d), angle (a).
        #  Xij: X of frame j relative to frame i.
        self.y15 = sum(model.body(f"link_{i}").pos[1] for i in [2,3,4,5])

        tb1 = model.body("link_1").pos
        mujoco.mju_quat2Mat((Rwb := np.empty((3,3))).reshape(9), model.body("base_link").quat)
        twb = model.body("base_link").pos
        self.tw1 = twb + Rwb @ tb1

        self.t45 = model.body("link_5").pos.copy()
        self.t5e = model.site("end_effector_site").pos.copy()

        self.t12 = model.body("link_2").pos.copy()

        x23, _, z23 = model.body("link_3").pos
        x34, _, z34 = model.body("link_4").pos
        self.dzx23 = np.linalg.norm([x23, z23])
        self.dzx34 = np.linalg.norm([x34, z34])
        self.azx23 = np.arctan2(x23, z23)
        self.azx34 = np.arctan2(x34, z34)


    def solve(self, ee_pos: np.ndarray, roll: float, pitch: float) -> tuple[np.ndarray, bool]:
        """Compute the analytical solution to the inverse kinematics of a "zyyyx" 5 dofs robot arm.
        The yaw of the end-effector (EE) is totally determined by the EE's position and roll, and
        therefore cannot be chosen freely.
        Arguments:
        - ee_pos (np.ndarray): desired position of the end-effector in world frame.
        - roll (float): desired roll angle of the end-effector in world frame.
        - pitch (float): desired pitch angle of the end-effector in world frame.
        Returns:
        - (np.ndarray): [j1,j2,j3,j4,j5] angles of the joints in radians solving the IK problem.
        - (bool): whether the pose is unreachable (e.g. ee_pos is too far).

        Note: there are two solutions, the one returned makes the elbow (joint 3) bend "inwards".

        Overview of the analytical derivation:
        0) Joint 5 angle is a function of the roll angle.
        1) Infer EE's yaw angle in world (W) frame from EE's position and roll (in W) and from the
        lateral offset of joint 5 from the robot's "sagittal plane" (the XZ plane of joint 1 frame
        i.e. the plane orthogonal to the axis of joints 2,3 and 4). An non-zero offset implies EE's
        yaw will not be exactly equal to the angle of joint 1 (around the vertical axis).
        2) Joint 1 angle is a function of the yaw angle.
        3) Infer the (x,z) coordinates of joint 4 in joint 1 frame from yaw, roll, EE's position.
        4) Infer the angles of joints 2 and 3 by solving a 2D IK problem in joint 1 (XZ) plane.
        4) Joint 4 angle is a function of the pitch and the angles of joints 2 and 3.
        """
        # Determine yaw angle.
        yaw = self.infer_yaw(twe := ee_pos, roll)
        
        # Determine the position of joint 4 in joint 1 frame.
        Rw1 = rot_axis('z', yaw + np.pi)
        t1e = Rw1.T @ (twe - self.tw1)

        R45 = rot_axis('x', -roll)
        t4e = self.t45 + R45 @ self.t5e

        R14 = rot_axis('y', -pitch)
        t14 = t1e - R14 @ t4e    # since t1e = t14 + R14 @ t4e

        # Determine the angles of joints 2 and 3.
        x, _, z = t14 - self.t12
        elbow = 'down' if x > 0 else 'up'
        angles, unreachable = solve_2D_IK(x, z, self.dzx23, self.dzx34, elbow=elbow)

        return self.get_KochV11_joint_angles(*angles[elbow], yaw, pitch, roll), unreachable


    def infer_yaw(self, twe, roll):
        _, y5e, z5e = self.t5e
        y1e = self.y15 + np.cos(roll) * y5e + np.sin(roll) * z5e

        x, y, _ = twe - self.tw1
        r, a = np.sqrt(x*x + y*y), np.arctan2(y, x)
        yaw = a + np.arcsin(y1e / r)

        return yaw


    def get_KochV11_joint_angles(self, alpha, beta, yaw, pitch, roll):
        j1 = np.pi/2 - yaw
        j2 = -alpha - self.azx23
        j3 = beta - self.azx23 + self.azx34
        j4 = -pitch - j2 + j3
        j5 = -roll

        return wrap_neg_pi_pi(np.array([j1, j2, j3, j4, j5]))


# ------------------------------- DAMPED LEAST SQUARES IK ------------------------------- #

def damped_least_squares_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_ee_pos: np.ndarray,
    ee_site: str | int = "end_effector_site",
    num_dof: int = 5,
    lm_damping: float = 0.001,
    tolerance_err: float = 0.001,
    max_iter: int = 20,
    extra_metrics: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Inverse Kinematics for a MuJoCo robotic arm to reach a target end effector position. Stops 
    when the error is below tolerance_err or max_iter is reached.

    Args:
        model (mujoco.MjModel): MuJoCo model object
        data (mujoco.MjData): MuJoCo data object
        target_ee_pos (np.ndarray): Target end effector position [x, y, z] in world frame.
        ee_site (str | int, optional): Name or id of the end effector site (default="end_effector_site").
        num_dof (int, optional): Number of degrees of freedom, gripper not included (default=5).
        lm_damping (float, optional): Damping factor for the regularized inverse computation (default=0.001).
        tolerance_err (float, optional): Tolerance error (default=0.001).
        max_iter (int, optional): Maximum number of iterations (default=20).
        extra_metrics (bool, optional): Whether to return extra metrics: success, error vector, iters (default=False).

    Returns:
        np.ndarray | tuple: Target joint positions, and optionally a dict of metrics if extra_metrics is True.
    """
    
    real_qpos = data.qpos.copy() # copy

    q = data.qpos[:num_dof] # view
    ee_id = ee_site if isinstance(ee_site, int) else model.site(ee_site).id
    ee_pos = data.site(ee_id).xpos # view (ee_pos = FK(q))
    jacp = np.empty((3, model.nv))
    J = jacp[:, :num_dof] # view

    for iter in range(max_iter):
        error = target_ee_pos - ee_pos
        
        if np.linalg.norm(error) < tolerance_err:
            break

        # Jacobian: J <- d(ee_pos)/dq
        mujoco.mj_jacSite(model, data, jacp, None, ee_id)

        # Damped least squares (Levenberg-Marquardt Algorithm): q <- q + Jᵀ(JJᵀ + λI)⁻¹e
        q += np.linalg.solve(J @ J.T + lm_damping * np.eye(3), error).dot(J)

        np.clip(q, *model.jnt_range[:num_dof].T, out=q)
        
        # Forward kinematics: ee_pos <- FK(q)
        mujoco.mj_fwdPosition(model, data)

    q_IK = q.copy() # copy

    # Restore the real qpos
    data.qpos = real_qpos
    mujoco.mj_fwdPosition(model, data)

    return q_IK if not extra_metrics else (q_IK, {"success": (np.linalg.norm(error) < tolerance_err), "error": error, "iters": iter})
