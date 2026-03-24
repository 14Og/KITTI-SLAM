from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pyscancontext as sc


@dataclass(frozen=True)
class LoopResult:
    """A verified loop closure between two frames."""

    idx_from: int
    idx_to: int
    T_relative: np.ndarray
    sc_dist: float
    icp_fitness: float
    icp_rmse: float


class LoopClosureDetector:
    """Scan Context loop closure detection with ICP verification."""

    def __init__(
        self,
        sc_threshold: float = 0.2,
        icp_fitness_threshold: float = 0.6,
        icp_rmse_threshold: float = 0.9,
        icp_max_distance: float = 5.0,
        voxel_size: float = 0.5,
        min_frame_gap: int = 100,
    ):
        self._sc = sc.SCManager()
        self._sc_threshold = sc_threshold
        self._icp_fitness_threshold = icp_fitness_threshold
        self._icp_rmse_threshold = icp_rmse_threshold
        self._icp_max_distance = icp_max_distance
        self._voxel_size = voxel_size
        self._min_frame_gap = min_frame_gap

        self._clouds: list[np.ndarray] = []

    def _yaw_matrix(self, yaw: float) -> np.ndarray:
        """4x4 rotation matrix for a yaw (Z-axis) rotation."""

        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[0, 0] = c
        T[0, 1] = -s
        T[1, 0] = s
        T[1, 1] = c
        return T

    def add_frame(self, cloud: np.ndarray) -> LoopResult | None:
        """Add a frame and check for loop closure.

        Args:
            cloud: (N, 3) point cloud in local sensor frame.

        Returns:
            LoopResult if a verified loop closure is found, None otherwise.
        """

        frame_idx = len(self._clouds)
        self._clouds.append(cloud)
        self._sc.add_node(cloud)

        loop_idx, sc_dist, yaw_diff = self._sc.detect_loop()

        if loop_idx == -1:
            return None
        if sc_dist > self._sc_threshold:
            return None
        if frame_idx - loop_idx < self._min_frame_gap:
            return None

        return self._verify_icp(frame_idx, loop_idx, yaw_diff, sc_dist)

    def _verify_icp(
        self, idx_from: int, idx_to: int, yaw_diff: float, sc_dist: float
    ) -> LoopResult | None:
        """Verify a loop candidate with ICP."""
        
        src_cloud = self._clouds[idx_from]
        tgt_cloud = self._clouds[idx_to]

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_cloud)
        src_pcd = src_pcd.voxel_down_sample(self._voxel_size)

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_cloud)
        tgt_pcd = tgt_pcd.voxel_down_sample(self._voxel_size)

        init_guess = self._yaw_matrix(yaw_diff)

        icp_result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            self._icp_max_distance,
            init_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )

        accepted = (
            icp_result.fitness >= self._icp_fitness_threshold
            and icp_result.inlier_rmse <= self._icp_rmse_threshold
        )
        if not accepted:
            return None

        return LoopResult(
            idx_from=idx_from,
            idx_to=idx_to,
            T_relative=np.array(icp_result.transformation),
            sc_dist=float(sc_dist),
            icp_fitness=icp_result.fitness,
            icp_rmse=icp_result.inlier_rmse,
        )
