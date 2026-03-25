import numpy as np
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP


class LiDARodometry:
    """KISS-ICP wrapper for frame-by-frame LiDAR odometry."""

    def __init__(self, max_range: float = 100.0, min_range: float = 5.0, deskew: bool = False):
        config = load_config(config_file=None, max_range=max_range)
        config.data.min_range = min_range
        config.data.deskew = deskew
        self._kiss = KissICP(config=config)
        self._poses: list[np.ndarray] = []

    def register(self, points: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
        """Register a (N, 3) point cloud and return the 4x4 pose in world frame."""

        assert isinstance(points, np.ndarray), f"expected ndarray, got {type(points)}"
        assert points.ndim == 2 and points.shape[1] == 3, f"expected (N, 3), got {points.shape}"

        if timestamps is None:
            timestamps = np.zeros(len(points))
        self._kiss.register_frame(points, timestamps)
        pose = self._kiss.last_pose.copy()
        self._poses.append(pose)
        return pose

    @property
    def poses(self) -> list[np.ndarray]:
        return self._poses

    @property
    def last_delta(self) -> np.ndarray:
        """Relative transform from the last registration (for odometry factors)."""

        return self._kiss.last_delta.copy()
