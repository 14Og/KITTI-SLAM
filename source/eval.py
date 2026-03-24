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
) -> dict[str, float]:
    """Compute RPE translation error (m) per frame."""
    
    traj_ref = poses_to_traj(gt_poses)
    traj_est = poses_to_traj(est_poses)
    rpe = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=metrics.Unit.frames,
    )
    rpe.process_data((traj_ref, traj_est))
    return rpe.get_all_statistics()


def compute_rpe_rotation(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    delta: float = 1.0,
) -> dict[str, float]:
    """Compute RPE rotation error (degrees) per frame."""
    
    traj_ref = poses_to_traj(gt_poses)
    traj_est = poses_to_traj(est_poses)
    rpe = metrics.RPE(
        metrics.PoseRelation.rotation_angle_deg,
        delta=delta,
        delta_unit=metrics.Unit.frames,
    )
    rpe.process_data((traj_ref, traj_est))
    return rpe.get_all_statistics()


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
    rpe_stats = compute_rpe(gt_poses, est_poses)
    rpe_rot_stats = compute_rpe_rotation(gt_poses, est_poses)

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
