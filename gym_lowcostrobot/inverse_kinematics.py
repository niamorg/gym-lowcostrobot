import mujoco
import numpy as np

def inverse_kinematics(
    model,
    data,
    ee_target_pos,
    ee_site="end_effector_site",
    num_dof=5,
    lm_damping=0.001,
    max_iter=20,
    tolerance_err=0.001,
    extra_metrics=False
):
    """
    Computes the inverse kinematics for a robotic arm to reach the target end effector position.

    :param ee_target_pos: numpy array of target end effector position [x, y, z]
    :param ee_site: str, name of the end effector site
    :param num_dof: int, number of degrees of freedom
    :param lm_damping: float, regularization factor for the pseudoinverse computation
    :param max_iter: int, maximum number of iterations
    :param tolerance_err: float, tolerance error
    :param extra_metrics: bool, whether to return extra metrics
    :return: numpy array of target joint positions, and a dict of metrics if extra_metrics is True
    """
    
    real_qpos = data.qpos.copy() # copy

    q = data.qpos[:num_dof] # view
    ee_id = model.site(ee_site).id
    ee_pos = data.site(ee_id).xpos # view (ee_pos = FK(q))
    jacp = np.empty((3, model.nv))
    J = jacp[:, :num_dof] # view

    for iter in range(max_iter):
        error = ee_target_pos - ee_pos
        
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

    return q_IK if not extra_metrics else (q_IK, {"success": (np.linalg.norm(error) < tolerance_err), "error": error, "iter": iter})