from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics
from evo.core.trajectory import PosePath3D
import scienceplots

plt.style.use(["grid", "notebook", "science"])


def poses_to_traj(poses: list[np.ndarray]) -> PosePath3D:
    """Convert a list of 4x4 SE(3) matrices to an evo PosePath3D."""
    
    return PosePath3D(poses_se3=poses)


def compute_ape(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
) -> tuple[dict[str, float], np.ndarray]:
    """Compute APE (translation). Returns (statistics, per-frame error array)."""
    
    traj_ref = poses_to_traj(gt_poses)
    traj_est = poses_to_traj(est_poses)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    return ape.get_all_statistics(), ape.error


def compute_rpe(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    delta: float = 1.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute RPE translation error (m) per frame. Returns (statistics, per-frame errors)."""

    traj_ref = poses_to_traj(gt_poses)
    traj_est = poses_to_traj(est_poses)
    rpe = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=metrics.Unit.frames,
    )
    rpe.process_data((traj_ref, traj_est))
    return rpe.get_all_statistics(), rpe.error


def compute_rpe_rotation(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    delta: float = 1.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute RPE rotation error (degrees) per frame. Returns (statistics, per-frame errors)."""

    traj_ref = poses_to_traj(gt_poses)
    traj_est = poses_to_traj(est_poses)
    rpe = metrics.RPE(
        metrics.PoseRelation.rotation_angle_deg,
        delta=delta,
        delta_unit=metrics.Unit.frames,
    )
    rpe.process_data((traj_ref, traj_est))
    return rpe.get_all_statistics(), rpe.error


def print_metrics(
    name: str, ape_stats: dict, rpe_stats: dict, rpe_rot_stats: Optional[dict] = None
):
    """Print APE and RPE summary for a named trajectory."""
    
    print(f"  {name}:")
    print(
        f"    APE       rmse={ape_stats['rmse']:.4f}m   mean={ape_stats['mean']:.4f}m   max={ape_stats['max']:.4f}m"
    )
    print(
        f"    RPE trans rmse={rpe_stats['rmse']:.4f}m   mean={rpe_stats['mean']:.4f}m   max={rpe_stats['max']:.4f}m"
    )
    if rpe_rot_stats is not None:
        print(
            f"    RPE rot   rmse={rpe_rot_stats['rmse']:.4f}°   mean={rpe_rot_stats['mean']:.4f}°   max={rpe_rot_stats['max']:.4f}°"
        )


def plot_metrics_over_time(
    gt_poses: list[np.ndarray],
    estimates: dict[str, list[np.ndarray]],
    title: str = "",
):
    """Plot APE, RPE translation, and RPE rotation vs frame for multiple methods.

    Args:
        gt_poses: ground truth poses.
        estimates: {method_name: list of 4x4 poses}.
        title: plot title prefix.
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    window = 50

    for name, est_poses in estimates.items():
        _, ape_err = compute_ape(gt_poses, est_poses)
        _, rpe_err = compute_rpe(gt_poses, est_poses)
        _, rpe_rot_err = compute_rpe_rotation(gt_poses, est_poses)

        axes[0].plot(ape_err, label=name, linewidth=1.0, alpha=0.85)

        # Raw RPE as faint background, running average as solid line
        kernel = np.ones(window) / window
        rpe_smooth = np.convolve(rpe_err, kernel, mode="valid")
        rpe_rot_smooth = np.convolve(rpe_rot_err, kernel, mode="valid")
        offset = window // 2

        color = axes[1].plot(rpe_err, linewidth=0.3, alpha=0.25)[0].get_color()
        axes[1].plot(np.arange(offset, offset + len(rpe_smooth)), rpe_smooth,
                     label=name, linewidth=1.5, color=color)

        axes[2].plot(rpe_rot_err, linewidth=0.3, alpha=0.25, color=color)
        axes[2].plot(np.arange(offset, offset + len(rpe_rot_smooth)), rpe_rot_smooth,
                     label=name, linewidth=1.5, color=color)

    axes[0].set_ylabel("APE (m)")
    axes[0].set_title(f"{title} — APE" if title else "APE")
    axes[0].legend()

    axes[1].set_ylabel("RPE trans (m)")
    axes[1].set_title("RPE translation")

    axes[2].set_ylabel("RPE rot (deg)")
    axes[2].set_title("RPE rotation")
    axes[2].set_xlabel("Frame")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_graph(
    poses: list[np.ndarray],
    loop_edges: list[tuple[int, int]],
    imu_edges: Optional[list[tuple[int, int]]] = None,
    gt_poses: Optional[list[np.ndarray]] = None,
    title: str = "Pose Graph",
    max_nodes: Optional[int] = None,
    start_node: Optional[int] = None,
):
    """Plot the pose graph spatially: nodes at XY positions, edges color-coded by type.

    Args:
        poses: optimized 4x4 poses (nodes).
        loop_edges: list of (i, j) loop closure pairs.
        imu_edges: list of (i, j) IMU factor pairs (M2 only).
        gt_poses: optional ground truth trajectory.
        title: plot title.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    # Clip to subgraph window — default to region around first loop closure
    window = max_nodes or 150
    if start_node is None:
        if loop_edges:
            first_loop = min(max(i, j) for i, j in loop_edges)
            start_node = max(0, first_loop - window // 4)
        else:
            start_node = 0
    end_node = min(len(poses), start_node + window)
    poses = poses[start_node:end_node]
    loop_edges = [(i - start_node, j - start_node)
                  for i, j in loop_edges if start_node <= i < end_node and start_node <= j < end_node]
    if imu_edges:
        imu_edges = [(i - start_node, j - start_node)
                     for i, j in imu_edges if start_node <= i < end_node and start_node <= j < end_node]
    if gt_poses is not None:
        gt_poses = gt_poses[start_node:end_node]

    xy = extract_xy(poses)
    _, ax = plt.subplots(figsize=(9, 9))

    # Ground truth
    if gt_poses is not None:
        gt_xy = extract_xy(gt_poses)
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "--", color="gray", linewidth=1.2,
                label="Ground truth", zorder=1)

    # Odometry factors (sequential edges)
    ax.plot(xy[:, 0], xy[:, 1], color="#4a90d9", linewidth=1.0,
            label="Odometry factors", zorder=2, alpha=0.7)

    # IMU factors — quadratic bezier arcs (LIO-SAM style), vectorized via LineCollection
    if imu_edges:
        avg_edge = np.linalg.norm(np.diff(xy, axis=0), axis=1).mean()
        arc_h = avg_edge * 1.5  # arc height = 1.5x average edge length

        idx_i = np.array([i for i, j in imu_edges])
        idx_j = np.array([j for i, j in imu_edges])
        p0 = xy[idx_i]
        p2 = xy[idx_j]

        # Control point: midpoint offset perpendicular to the edge (consistent side)
        edge = p2 - p0
        perp = np.stack([-edge[:, 1], edge[:, 0]], axis=1)
        norm = np.linalg.norm(perp, axis=1, keepdims=True)
        norm = np.where(norm < 1e-10, 1.0, norm)
        p1 = (p0 + p2) / 2 + arc_h * perp / norm

        # Sample quadratic bezier: shape (n_edges, 12, 2)
        t = np.linspace(0, 1, 12)[None, :, None]
        arcs = (1 - t)**2 * p0[:, None, :] + 2 * (1 - t) * t * p1[:, None, :] + t**2 * p2[:, None, :]

        lc = LineCollection(arcs, color="#d47324", linewidth=0.7, alpha=0.5, zorder=3)
        ax.add_collection(lc)
        ax.plot([], [], color="#d47324", linewidth=2.0, alpha=0.9,
                label=f"IMU factors ({len(imu_edges)})")

    # Loop closure factors — curved arcs to avoid dense polygon fill
    arc_rad = 0.25
    for i, j in loop_edges:
        patch = FancyArrowPatch(
            posA=(xy[i, 0], xy[i, 1]),
            posB=(xy[j, 0], xy[j, 1]),
            connectionstyle=f"arc3,rad={arc_rad}",
            color="crimson",
            linewidth=0.7,
            alpha=0.25,
            arrowstyle="-",
            zorder=4,
        )
        ax.add_patch(patch)
    if loop_edges:
        ax.plot([], [], color="crimson", linewidth=2.0,
                label=f"Loop closure factors ({len(loop_edges)})")

    # Nodes — subsample for large graphs, but keep them visible
    node_step = max(1, len(xy) // 400)
    ax.scatter(xy[::node_step, 0], xy[::node_step, 1],
               s=14, color="#1a1a2e", zorder=6, linewidths=0,
               label=f"Nodes ({len(xy)})")

    # ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_xy(poses: list[np.ndarray]) -> np.ndarray:
    """Extract (N, 2) XY positions from 4x4 poses."""
    
    return np.array([[p[0, 3], p[1, 3]] for p in poses])


def plot_trajectories(
    trajectories: dict[str, list[np.ndarray]],
    gt_poses: Optional[list[np.ndarray]] = None,
    title: str = "",
    styles: Optional[dict[str, dict]] = None,
):
    """Plot multiple trajectories on one figure.

    Args:
        trajectories: {name: list of 4x4 poses} for each method.
        gt_poses: optional ground truth poses.
        title: plot title.
        styles: optional {name: dict of matplotlib kwargs} per trajectory.
    """
    
    default_styles = {}
    if styles:
        default_styles.update(styles)

    _, ax = plt.subplots(figsize=(8, 8))

    if gt_poses is not None:
        gt_xy = extract_xy(gt_poses)
        ax.plot(
            gt_xy[:, 0],
            gt_xy[:, 1],
            label="Ground truth",
            linewidth=1.5,
            linestyle="--",
            color="gray",
        )

    for name, poses in trajectories.items():
        xy = extract_xy(poses)
        style = default_styles.get(name, {})
        ax.plot(xy[:, 0], xy[:, 1], label=name, linewidth=1.5, **style)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ape_on_trajectory(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    title: str = "APE mapped onto trajectory",
):
    """Plot estimated trajectory colored by APE, with metrics in a text box."""
    
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    ape_stats, ape_errors = compute_ape(gt_poses, est_poses)
    rpe_stats, _ = compute_rpe(gt_poses, est_poses)
    rpe_rot_stats, _ = compute_rpe_rotation(gt_poses, est_poses)

    est_xy = extract_xy(est_poses)
    gt_xy = extract_xy(gt_poses)

    segments = [[est_xy[i], est_xy[i + 1]] for i in range(len(est_xy) - 1)]
    norm = Normalize(vmin=float(ape_errors.min()), vmax=float(ape_errors.max()))
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=2)
    lc.set_array(ape_errors[:-1])

    metrics_text = (
        f"APE       rmse={ape_stats['rmse']:.3f}m  mean={ape_stats['mean']:.3f}m\n"
        f"RPE trans rmse={rpe_stats['rmse']:.3f}m  mean={rpe_stats['mean']:.3f}m\n"
        f"RPE rot   rmse={rpe_rot_stats['rmse']:.3f}°  mean={rpe_rot_stats['mean']:.3f}°"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "--", color="gray", label="Ground truth", linewidth=1.5)
    ax.add_collection(lc)
    ax.autoscale_view()
    fig.colorbar(lc, ax=ax, label="APE (m)", fraction=0.046, pad=0.04)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.02,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        family="monospace",
    )
    plt.tight_layout()
    plt.show()
