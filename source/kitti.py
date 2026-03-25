from datetime import datetime
from typing import Iterator

import numpy as np
import pykitti


class KITTIDataset:
    """Wrapper around pykitti for LiDAR odometry sequences."""

    def __init__(self, basedir: str, sequence: str):
        self.dataset = pykitti.odometry(basedir, sequence)
        self.T_cam0_velo = self.dataset.calib.T_cam0_velo

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yields (N, 3) point clouds in velodyne frame."""
        for i in range(len(self)):
            yield self.get_cloud(i)

    def get_cloud(self, idx: int) -> np.ndarray:
        """Load a single scan as (N, 3) xyz points."""
        return self.dataset.get_velo(idx)[:, :3]

    @property
    def gt_poses(self) -> list[np.ndarray] | None:
        """Ground truth poses in velodyne frame (list of 4x4 matrices).

        pykitti provides GT in cam0 frame; we transform to velodyne.
        Returns None if GT is not available (sequences 11-21).
        """
        if not self.dataset.poses:
            return None
        T_velo_cam0 = np.linalg.inv(self.T_cam0_velo)
        return [T_velo_cam0 @ T @ self.T_cam0_velo for T in self.dataset.poses]


class KITTIRawIMU:
    """Wrapper around pykitti.raw for IMU data access."""

    # Mapping from KITTI odometry sequence to raw drive (date, drive_number)
    IMU_LiDAR_map: dict[str, tuple[str, str]] = {
        "00": ("2011_10_03", "0027"),
        "06": ("2011_09_30", "0020"),
        "07": ("2011_09_30", "0027"),
    }

    def __init__(self, raw_basedir: str, sequence: str):
        if sequence not in self.IMU_LiDAR_map:
            raise ValueError(
                f"No raw mapping for odometry sequence {sequence}. "
                f"Available: {list(self.IMU_LiDAR_map.keys())}"
            )
        date, drive = self.IMU_LiDAR_map[sequence]
        self._raw = pykitti.raw(raw_basedir, date, drive, dataset="extract")
        self._raw_basedir = raw_basedir
        self._date = date
        self._drive = drive

        self._t0 = self._raw.timestamps[0]
        self._timestamps_sec = self._parse_oxts_timestamps()
        self._velo_timestamps_sec = self._parse_velo_timestamps(raw_basedir, date, drive)

    def _parse_oxts_timestamps(self) -> np.ndarray:
        """Convert pykitti OXTS datetime timestamps to float seconds from t0."""

        return np.array([(t - self._t0).total_seconds() for t in self._raw.timestamps])

    def _parse_velo_timestamps(self, raw_basedir: str, date: str, drive: str) -> np.ndarray:
        """Parse velodyne timestamps.txt to float seconds from t0."""

        import os

        ts_file = os.path.join(
            raw_basedir,
            date,
            f"{date}_drive_{drive}_extract",
            "velodyne_points",
            "timestamps.txt",
        )
        velo_times = []
        with open(ts_file) as f:
            for line in f:
                s = line.strip()[:26]  # truncate nanoseconds to microseconds
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
                velo_times.append((dt - self._t0).total_seconds())
        return np.array(velo_times)

    @property
    def timestamps(self) -> np.ndarray:
        """IMU (OXTS) timestamps in seconds from first IMU frame. Shape (N,)."""
        return self._timestamps_sec

    @property
    def velo_timestamps(self) -> np.ndarray:
        """Velodyne frame timestamps in seconds from first IMU frame. Shape (M,)."""
        return self._velo_timestamps_sec

    @property
    def T_velo_imu(self) -> np.ndarray:
        """4x4 transform from IMU frame to Velodyne frame."""
        return self._raw.calib.T_velo_imu

    def get_imu_between(
        self, t_start: float, t_end: float
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Get IMU measurements between two timestamps.

        Args:
            t_start: Start time in seconds from first IMU frame.
            t_end: End time in seconds from first IMU frame.

        Returns:
            List of (timestamp_sec, accel_xyz, gyro_xyz) tuples.
            accel_xyz: (3,) array [ax, ay, az] in m/s².
            gyro_xyz: (3,) array [wx, wy, wz] in rad/s.
        """

        mask = (self._timestamps_sec >= t_start) & (self._timestamps_sec <= t_end)
        indices = np.where(mask)[0]

        measurements = []
        for idx in indices:
            p = self._raw.oxts[idx].packet
            accel = np.array([p.ax, p.ay, p.az])
            gyro = np.array([p.wx, p.wy, p.wz])
            measurements.append((self._timestamps_sec[idx], accel, gyro))

        return measurements

    def get_lidar_timestamps(self, n_frames: int) -> np.ndarray:
        """Get velodyne timestamps for each odometry frame.

        KITTI odometry sequences start at raw frame 0 (per devkit mapping).
        Uses the parsed raw velodyne timestamps directly.
        """

        n_raw = len(self._velo_timestamps_sec)
        if n_frames > n_raw:
            raise ValueError(f"Need {n_frames} frames but raw has only {n_raw} velodyne timestamps")
        return self._velo_timestamps_sec[:n_frames]
