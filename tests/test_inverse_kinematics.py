import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mujoco.viewer
from time import sleep

from gym_lowcostrobot.inverse_kinematics import solve_2D_IK, InverseKinematicsZYYYX, damped_least_squares_ik

# --------------------------------- TEST ANALYTICAL IK ---------------------------------------------
def test_solve_2D_IK():
    x, y, l1, l2 = 1.5, 1, 2, 3
    sols, _ = solve_2D_IK(x, y, l1, l2, elbow='both')
    for alpha, beta in sols.values():
        phi_1 = np.pi/2 + alpha
        phi_2 = np.pi/2 + alpha + beta
        x_pred = l1 * np.cos(phi_1) + l2 * np.cos(phi_2)
        y_pred = l1 * np.sin(phi_1) + l2 * np.sin(phi_2)
        assert (x_pred - x)**2 + (y_pred - y)**2 < 1e-6, f"solve_IK_2D() failed: x_pred: {x_pred}, y_pred: {y_pred}, x: {x}, y: {y}"
    print("test for solve_IK_2D() passed")


def test_InverseKinematicsZYYYX(viz=False):
    model = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/low_cost_robot_6dof/arm_and_balls_scene.xml")    
    ik = InverseKinematicsZYYYX(model)
    desired_ee_pos, desired_roll, desired_pitch = np.array([0.02, 0.2, 0.1]), 0, 1.
    j, _ = ik.solve(desired_ee_pos, desired_roll, desired_pitch)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[:5] = j
    mujoco.mj_forward(model, data)

    ee_pos = data.site("end_effector_site").xpos
    assert np.all(np.abs(ee_pos - desired_ee_pos) < 1e-6), f"test InverseKinematicsZYYYX failed: ee_pos: {ee_pos}, desired_ee_pos: {desired_ee_pos}"

    
    R = data.site("end_effector_site").xmat.reshape(3,3).copy()
    roll, pitch, yaw = Rotation.from_matrix(R).as_euler('xyz')
    assert np.abs(roll - desired_roll) < 1e-6, f"test InverseKinematicsZYYYX failed: roll: {roll}, desired_roll: {desired_roll}"
    assert np.abs(pitch - desired_pitch) < 1e-6, f"test InverseKinematicsZYYYX failed: pitch: {pitch}, desired_pitch: {desired_pitch}"

    print("test InverseKinematicsZYYYX passed")

    if viz:
        viewer = mujoco.viewer.launch_passive(model, data)
        mujoco.mj_resetData(model, data)
        model.body("colored_frame").pos = desired_ee_pos
        model.geom("red_ball").pos = desired_ee_pos

        data.ctrl[:5] = j        
        for _ in range(1000):
            viewer.sync()
            mujoco.mj_step(model, data)
            sleep(0.01)


if __name__ == "__main__":
    passed = True
    try:
        test_solve_2D_IK()
    except Exception as e:
        print(f"Error in test_solve_2D_IK: {e}")
        passed = False
    try:
        test_InverseKinematicsZYYYX(viz=False)
    except Exception as e:
        print(f"Error in test_InverseKinematicsZYYYX: {e}")
        passed = False
    
    print(f"{'='*50}")
    if passed:
        print("✓ ANALYTICAL IK: All tests passed")
    else:
        print("✗ ANALYTICAL IK: Some tests failed")
    print(f"{'='*50}\n")

    sleep(1)

# --------------------------------- END TEST ANALYTICAL IK ---------------------------------------------



# --------------------------------- TEST DAMPED LEAST SQUARES IK ---------------------------------------------

def set_home(model: mujoco.MjModel, data: mujoco.MjData):
    mujoco.mj_resetData(model, data)
    # data.qpos[:] = model.keyframe("home").qpos
    # data.qvel[:] = model.keyframe("home").qvel
    # data.ctrl[:] = model.keyframe("home").ctrl
    mujoco.mj_forward(model, data)


# Test (LLM generated): Check that IK has no side effects on the 
# simulation state and that it is correct.
def test_damped_least_squares_ik():
    print("Testing Inverse Kinematics...")
    
    # 0) Instantiate model and data, call mj_forward
    model_path = "gym_lowcostrobot/assets/low_cost_robot_6dof/follower.xml"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    set_home(model, data)
    
    # Store initial state (full mujoco simulation state)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    initial_qacc = data.qacc.copy()
    initial_ctrl = data.ctrl.copy()
    initial_qfrc_applied = data.qfrc_applied.copy()
    initial_xfrc_applied = data.xfrc_applied.copy()
    initial_ee_pos = data.site("end_effector_site").xpos.copy()
    
    print(f"Initial end effector position: {initial_ee_pos}")
    print(f"Initial joint positions: {initial_qpos[:6]}")
    
    # Test targets
    test_targets = [
        np.array([0.2, 0.0, 0.3]),   # Forward and up
        np.array([0.0, 0.2, 0.2]),   # Right and up
        np.array([-0.1, 0.0, 0.25]), # Back and up
    ]
    
    all_tests_passed = True
    
    for i, target_pos in enumerate(test_targets):
        print(f"\n--- Test {i+1}: Target {target_pos} ---")
        
        # Store full state before IK
        qpos_before = data.qpos.copy()
        qvel_before = data.qvel.copy()
        qacc_before = data.qacc.copy()
        ctrl_before = data.ctrl.copy()
        qfrc_applied_before = data.qfrc_applied.copy()
        xfrc_applied_before = data.xfrc_applied.copy()
        ee_pos_before = data.site("end_effector_site").xpos.copy()
        
        # 1) Call IK
        try:
            ik_solution, metrics = damped_least_squares_ik(
                model=model,
                data=data,
                target_ee_pos=target_pos,
                ee_site="end_effector_site",
                num_dof=6,
                extra_metrics=True
            )
            
            # 2) Check that the full mujoco state in data is preserved
            qpos_after = data.qpos.copy()
            qvel_after = data.qvel.copy()
            qacc_after = data.qacc.copy()
            ctrl_after = data.ctrl.copy()
            qfrc_applied_after = data.qfrc_applied.copy()
            xfrc_applied_after = data.xfrc_applied.copy()
            ee_pos_after = data.site("end_effector_site").xpos.copy()
            
            state_preserved = (
                np.array_equal(qpos_before, qpos_after) and
                np.array_equal(qvel_before, qvel_after) and
                np.array_equal(qacc_before, qacc_after) and
                np.array_equal(ctrl_before, ctrl_after) and
                np.array_equal(qfrc_applied_before, qfrc_applied_after) and
                np.array_equal(xfrc_applied_before, xfrc_applied_after) and
                np.array_equal(ee_pos_before, ee_pos_after)
            )
            
            print(f"Full mujoco state preserved: {state_preserved}")
            
            # 3) Check IK is correct
            # Apply IK solution and verify forward kinematics
            data.qpos[:6] = ik_solution
            mujoco.mj_forward(model, data)
            achieved_pos = data.site("end_effector_site").xpos
            achieved_error = np.linalg.norm(target_pos - achieved_pos)
            
            print(f"IK solution: {ik_solution}")
            print(f"Target: {target_pos}")
            print(f"Achieved: {achieved_pos}")
            print(f"Error: {achieved_error:.6f}")
            print(f"Success: {metrics['success']}")
            
            ik_correct = metrics['success'] and achieved_error < 0.002

            # Restore full state for next test
            data.qpos = qpos_before
            data.qvel = qvel_before
            data.qacc = qacc_before
            data.ctrl = ctrl_before
            data.qfrc_applied = qfrc_applied_before
            data.xfrc_applied = xfrc_applied_before
            mujoco.mj_forward(model, data)
            
            test_passed = state_preserved and ik_correct
            if test_passed:
                print("✓ Test PASSED")
            else:
                print("✗ Test FAILED")
                all_tests_passed = False
                
        except Exception as e:
            print(f"✗ Test FAILED with exception: {e}")
            all_tests_passed = False
    
    # Final verification of full state
    final_qpos = data.qpos.copy()
    final_qvel = data.qvel.copy()
    final_qacc = data.qacc.copy()
    final_ctrl = data.ctrl.copy()
    final_qfrc_applied = data.qfrc_applied.copy()
    final_xfrc_applied = data.xfrc_applied.copy()
    final_ee_pos = data.site("end_effector_site").xpos.copy()
    
    final_state_preserved = (
        np.array_equal(initial_qpos, final_qpos) and
        np.array_equal(initial_qvel, final_qvel) and
        np.array_equal(initial_qacc, final_qacc) and
        np.array_equal(initial_ctrl, final_ctrl) and
        np.array_equal(initial_qfrc_applied, final_qfrc_applied) and
        np.array_equal(initial_xfrc_applied, final_xfrc_applied) and
        np.array_equal(initial_ee_pos, final_ee_pos)
    )
    
    print(f"\n--- Final State Verification ---")
    print(f"Full mujoco state preserved: {final_state_preserved}")
    
    if not final_state_preserved:
        all_tests_passed = False
    
    # Summary
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("✓ ALL TESTS PASSED! Inverse kinematics is working correctly.")
    else:
        print("✗ SOME TESTS FAILED! Check the implementation.")
    print(f"{'='*50}")


# Thorough test of IK correctness, which includes a study on the time it takes the robot
# to converge to the target position, with plots and optional visualization.
def test_ik_time_to_convergence_with_plots_and_viz(action_ranges=[0.03], timestep=0.002, viz=False):
    print("\n=== IK Time to Convergence with Plot ===")
    import numpy as np
    import matplotlib.pyplot as plt
    import mujoco.viewer
    import time

    model = mujoco.MjModel.from_xml_path("/home/romain/github/gym-lowcostrobot/gym_lowcostrobot/assets/low_cost_robot_6dof/arm_and_balls_scene.xml")
    data = mujoco.MjData(model)
    set_home(model, data)

    model.opt.timestep = timestep

    if viz:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.azimuth = -65.0
        viewer.cam.distance = 0.8
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.0])
    
    rng = np.random.default_rng(2)

    ee_site = "end_effector_site"
    n_init_positions = 10
    n_samples_per_pos = 100
    n_steps = round(1 / model.opt.timestep) // 2
    tolerances = [0.002, 0.005] #, 0.01]  # 2mm, 5mm, 10mm

    # Store results for all action ranges
    all_results = {}
    
    for action_range in action_ranges:
        print(f"\n=== Processing action_range = {action_range} ===")
        radius = action_range
        all_errors = []
        
        for pos_idx in range(n_init_positions):
            sampled_ee_pos = rng.uniform(low=[-0.15, 0.05, 0.0], high=[0.15, 0.20, 0.30], size=3)
            random_joint_vel = rng.uniform(low=-2.0, high=2.0, size=5)
            
            print(f"\n--- Initial Position {pos_idx+1} {sampled_ee_pos=} {random_joint_vel=}---")
            print(f"distance sampled_ee_pos to default ee_pos: {np.linalg.norm(data.site(ee_site).xpos.copy() - sampled_ee_pos)}")

            joint_positions, metrics = damped_least_squares_ik(model, data, sampled_ee_pos, extra_metrics=True, max_iter=1000, tolerance_err=0.05, lm_damping=0.1)
            if not metrics["success"]:
                print(f"Initial position {pos_idx+1} IK failed: {metrics}, error: {np.linalg.norm(metrics['error'])}")
                continue

            for sample_idx in range(n_samples_per_pos):
                set_home(model, data)
                data.qpos[:] = np.append(joint_positions, 0.0)
                data.qvel[:] = np.append(random_joint_vel, 0.0)
                mujoco.mj_forward(model, data)
                initial_ee_pos = data.site(ee_site).xpos.copy()

                if viz and sample_idx == 0:
                    model.geom("red_ball").pos[:] = initial_ee_pos
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(model.opt.timestep)

                while True:
                    while np.linalg.norm(offset := rng.uniform(-radius, radius, size=3)) > radius: pass
                    sampled_ee_pos = initial_ee_pos + offset
                
                    # Run IK to get target joint positions
                    target_joint_pos, metrics = damped_least_squares_ik(model, data, sampled_ee_pos, extra_metrics=True)
                    if not metrics["success"]:
                        print(f"    [Sample {sample_idx+1}] IK failed: {metrics["error"].round(4)}, error: {np.linalg.norm(metrics['error'])} offset:{np.linalg.norm(offset)}")
                    ik_ee_pos = sampled_ee_pos - metrics["error"]
                    break

                # update position of yellow ball
                if viz and sample_idx % 10 == 0:
                    model.geom("yellow_ball").pos[:] = ik_ee_pos
                    mujoco.mj_forward(model, data)
                    viewer.sync()

                # Simulate for 1000 steps
                data.ctrl[:] = np.append(target_joint_pos, 0.0)
                
                errors = []
                for t in range(n_steps):
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if viz and sample_idx % 10 == 0:
                        model.geom("green_ball").pos[:] = data.site(ee_site).xpos.copy()
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        time.sleep(model.opt.timestep)
                    # print("data.qvel", data.qvel.round(2))
                    ee_now = data.site(ee_site).xpos.copy()
                    err = np.linalg.norm(ee_now - ik_ee_pos)
                    errors.append(err)
                
                all_errors.append(np.array(errors))

        all_errors = np.array(all_errors)  # shape: [n_runs, n_steps]
        print(f"Action range {action_range}: {all_errors.shape}")
        n_runs = all_errors.shape[0]

        # Compute scores for this action range
        for tolerance in tolerances:
            n_success = np.sum(np.any(all_errors < tolerance, axis=1))
            print(f"Action range {action_range}: Total runs: {n_runs}, Successes (error < {tolerance*1000:.1f}mm): {n_success} ({n_success/n_runs*100:.1f}%)")

        # Store results for this action range
        all_results[action_range] = all_errors

    # Create plots with all action ranges
    time_ms = np.arange(n_steps) * model.opt.timestep * 1000

    # Plot 1: Percentage of runs within tolerance
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(action_ranges)))  # Generate colors for each action range
    
    for i, action_range in enumerate(action_ranges):
        all_errors = all_results[action_range]
        color = colors[i]
        
        for tolerance in tolerances:
            # For each time step, proportion of runs that have reached within tolerance
            reached = (all_errors < tolerance)  # shape: [n_runs, n_steps]
            prop_reached = np.mean(reached, axis=0)  # [n_steps]
            
            # Choose line style: dotted for 5mm, full for 2mm
            linestyle = '--' if tolerance == 0.005 else '-'
            
            plt.plot(time_ms, 100 * prop_reached, 
                    linestyle=linestyle,
                    color=color,
                    label=f"Action range {100*action_range:.0f}cm, tolerance={tolerance*1000:.0f}mm")
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Percentage of runs within tolerance")
    plt.title(f"% of runs that reached target vs. time (timestep={timestep*1000}ms)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot 2: Average error
    plt.figure(figsize=(10, 6))
    for action_range in action_ranges:
        all_errors = all_results[action_range]
        plt.plot(time_ms, 1000 * all_errors.mean(axis=0), 
                label=f"Action range {100*action_range:.0f}cm")
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Average error (mm)")
    plt.title(f"Average error vs. time (timestep={timestep*1000}ms)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    print("\n=== End of IK Reachability with Plot ===\n")
    if viz:
        viewer.close()

if __name__ == "__main__":
    test_damped_least_squares_ik()
    test_ik_time_to_convergence_with_plots_and_viz(action_ranges=[0.01, 0.03, 0.05], viz=False)
    plt.show()