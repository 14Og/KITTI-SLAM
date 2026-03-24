import gtsam
import numpy as np

from dataclasses import dataclass, field


def pose3_from_matrix(T: np.ndarray) -> gtsam.Pose3:
    """Convert a 4x4 SE(3) matrix to gtsam.Pose3."""
    
    return gtsam.Pose3(T)


def matrix_from_pose3(pose: gtsam.Pose3) -> np.ndarray:
    """Convert gtsam.Pose3 to a 4x4 SE(3) matrix."""
    
    return pose.matrix()


@dataclass(frozen=True)
class NoiseSigmas:
    """Diagonal noise sigmas for each factor type (rx, ry, rz, tx, ty, tz)."""

    prior: np.ndarray = field(
        default_factory=lambda: np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    )
    odometry: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]))
    loop: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]))


class PoseGraph:
    """GTSAM ISAM2-based pose graph for LiDAR SLAM."""

    def __init__(self, noise: NoiseSigmas):
        params = gtsam.ISAM2Params()
        self._isam = gtsam.ISAM2(params)

        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(noise.prior)
        self._odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(noise.odometry)
        self._loop_noise = gtsam.noiseModel.Diagonal.Sigmas(noise.loop)

        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._n_nodes = 0

    def add_first_pose(self, pose: np.ndarray):
        """Add the first node with a prior factor anchoring it."""

        assert self._n_nodes == 0, "first pose already added"
        p = pose3_from_matrix(pose)
        self._graph.add(gtsam.PriorFactorPose3(0, p, self._prior_noise))
        self._initial.insert(0, p)
        self._n_nodes = 1

    def add_odometry(self, pose: np.ndarray, delta: np.ndarray):
        """Add a new node with an odometry factor from the previous node.

        Args:
            pose: 4x4 global pose estimate (initial guess for this node).
            delta: 4x4 relative transform from previous node to this node.
        """

        assert self._n_nodes > 0, "call add_first_pose first"
        i = self._n_nodes
        self._graph.add(
            gtsam.BetweenFactorPose3(i - 1, i, pose3_from_matrix(delta), self._odometry_noise)
        )
        self._initial.insert(i, pose3_from_matrix(pose))
        self._n_nodes += 1

    def add_loop_closure(self, i: int, j: int, T_relative: np.ndarray):
        """Add a loop closure factor between nodes i and j."""

        self._graph.add(
            gtsam.BetweenFactorPose3(i, j, pose3_from_matrix(T_relative), self._loop_noise)
        )

    def optimize(self) -> list[np.ndarray]:
        """Run ISAM2 update and return all optimized poses as 4x4 matrices."""

        self._isam.update(self._graph, self._initial)
        # Extra iterations for convergence
        self._isam.update()
        result = self._isam.calculateEstimate()

        # Clear incremental buffers (already absorbed by ISAM2)
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()

        return [matrix_from_pose3(result.atPose3(i)) for i in range(self._n_nodes)]

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def dot(self, first_n: int | None = None) -> str:
        """Return graphviz DOT string for the graph.

        Uses ISAM2 internals if already optimized, otherwise the pending buffer.
        If first_n is set, builds a subgraph with only the first N nodes.
        """

        # Use ISAM2 graph if available, otherwise the pending buffer
        isam_graph = self._isam.getFactorsUnsafe()
        if isam_graph.size() > 0:
            full_graph = isam_graph
            values = self._isam.calculateEstimate()
        else:
            full_graph = self._graph
            values = self._initial

        if first_n is None:
            return full_graph.dot(values)

        # Build subgraph with only factors touching nodes < first_n
        sub_graph = gtsam.NonlinearFactorGraph()
        sub_values = gtsam.Values()

        for i in range(min(first_n, self._n_nodes)):
            if values.exists(i):
                sub_values.insert(i, values.atPose3(i))

        for k in range(full_graph.size()):
            factor = full_graph.at(k)
            if factor is None:
                continue
            keys = factor.keys()
            if all(key < first_n for key in keys):
                sub_graph.add(factor)

        return sub_graph.dot(sub_values)
