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
