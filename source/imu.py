import numpy as np


def integrate_gyro(
    imu_measurements: list[tuple[float, np.ndarray, np.ndarray]],
    T_velo_imu: np.ndarray,
) -> np.ndarray:
    """Integrate gyroscope to get relative rotation in velodyne frame.

    Bypasses GTSAM preintegration — no gravity, no velocity, no bodyPSensor.
    Just direct Rodrigues integration of angular velocity.

    Returns:
        (3, 3) rotation matrix: relative rotation in velodyne body frame.
    """

    R_vi = T_velo_imu[:3, :3]
    R = np.eye(3)

    for k in range(1, len(imu_measurements)):
        t_prev, _, _ = imu_measurements[k - 1]
        t_curr, _, gyro = imu_measurements[k]
        dt = t_curr - t_prev
        if dt <= 0:
            continue

        # Transform gyro from IMU frame to velodyne frame
        omega_velo = R_vi @ gyro
        dtheta = omega_velo * dt
        angle = np.linalg.norm(dtheta)
        if angle > 1e-10:
            axis = dtheta / angle
            K = np.array(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ]
            )
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        else:
            dR = np.eye(3)

        R = R @ dR

    return R
