import numpy as np

def wrap_neg_pi_pi(x):
    return ((x + np.pi) % (2*np.pi)) - np.pi


def rotmat_to_eulers(R: np.ndarray) -> np.ndarray:
    """
    Returns the unique Euler angles (roll, pitch, yaw) corresponding to the rotation matrix R,
    such that: R = R_z(yaw) @ R_y(pitch) @ R_x(roll) and yaw, roll ∈ [-π, π], pitch ∈ [-π/2, π/2].
    Gimbal lock case is handled by setting roll = 0.
    """
    pitch = np.arcsin(-R[2,0])
    
    if not np.pi/2 - np.abs(pitch) < 1e-6:
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        # gimbal lock case
        roll = 0
        yaw = np.arctan2(-R[0,1], R[1,1])
    return np.array([roll, pitch, yaw])


def rot_axis(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([
            [1, 0, 0], 
            [0, c,-s], 
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return np.array([
            [c,-s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}")




def test_mat_to_eulers():
    rng = np.random.default_rng(1)
    for roll, pitch, yaw in rng.uniform([-np.pi, -np.pi/2, -np.pi], [np.pi, np.pi/2, np.pi], (100, 3)):
        R = rot_axis('z', yaw) @ rot_axis('y', pitch) @ rot_axis('x', roll)
        r, p, y = rotmat_to_eulers(R)
        assert np.allclose(r, roll, atol=1e-6)
        assert np.allclose(p, pitch, atol=1e-6)
        assert np.allclose(y, yaw, atol=1e-6)
    
    roll, pitch, yaw = np.pi/5, np.pi/2, np.pi/3
    R = rot_axis('z', yaw) @ rot_axis('y', pitch) @ rot_axis('x', roll)
    r, p, y = rotmat_to_eulers(R)
    assert r == 0
    assert p == np.pi/2

    print("All tests passed")


def test_gimbal_lock_plot():
    import matplotlib.pyplot as plt

    roll_true = np.pi / 4   # 45 degrees
    yaw_true = np.pi / 3    # 60 degrees
    # Generate pitches very close to 90 degrees (from 89 to 91 degrees)
    pitches = np.linspace(np.deg2rad(89), np.deg2rad(91), 200)
    rolls = []
    yaws = []

    for pitch in pitches:
        R = rot_axis('z', yaw_true) @ rot_axis('y', pitch) @ rot_axis('x', roll_true)
        r, p, y = rotmat_to_eulers(R)
        rolls.append(r)
        yaws.append(y)

    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(np.rad2deg(pitches), np.rad2deg(rolls), label='Recovered Roll')
    plt.axhline(np.rad2deg(roll_true), color='r', linestyle='--', label='True Roll')
    plt.ylabel('Roll (deg)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.rad2deg(pitches), np.rad2deg(yaws), label='Recovered Yaw')
    plt.axhline(np.rad2deg(yaw_true), color='r', linestyle='--', label='True Yaw')
    plt.xlabel('Pitch (deg)')
    plt.ylabel('Yaw (deg)')
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_mat_to_eulers()
    test_gimbal_lock_plot()