"""KITTI Graph-SLAM — Milestone 1 (LiDAR-only) and Milestone 2 (LiDAR-IMU).

Usage:
    uv run main.py --milestone 1 --sequence 07
    uv run main.py --milestone 2 --sequence 07
"""

import argparse

import numpy as np
from tqdm import tqdm

from source.eval import (
    compute_ape,
    compute_rpe,
    compute_rpe_rotation,
    plot_ape_on_trajectory,
    plot_graph,
    plot_metrics_over_time,
    plot_trajectories,
    print_metrics,
)
from source.graph import NoiseSigmas, PoseGraph
from source.imu import integrate_gyro
from source.kiss import LiDARodometry
from source.kitti import KITTIDataset, KITTIRawIMU
from source.loop_closure import LoopClosureDetector


def evaluate(
    sequence: str,
    gt_poses: list[np.ndarray],
    kiss_poses: list[np.ndarray],
    optimized: list[np.ndarray],
    graph: "PoseGraph",
    label: str = "ISAM2 + loop closures",
):
    """Compute and display metrics, plot trajectories and pose graph."""

    kiss_ape, _ = compute_ape(gt_poses, kiss_poses)
    kiss_rpe, _ = compute_rpe(gt_poses, kiss_poses)
    kiss_rpe_rot, _ = compute_rpe_rotation(gt_poses, kiss_poses)

    opt_ape, _ = compute_ape(gt_poses, optimized)
    opt_rpe, _ = compute_rpe(gt_poses, optimized)
    opt_rpe_rot, _ = compute_rpe_rotation(gt_poses, optimized)

    print(f"\nSequence {sequence} — results:")
    print_metrics("KISS-ICP (raw)", kiss_ape, kiss_rpe, kiss_rpe_rot)
    print_metrics(label, opt_ape, opt_rpe, opt_rpe_rot)

    plot_trajectories(
        trajectories={"KISS-ICP": kiss_poses, label: optimized},
        gt_poses=gt_poses,
        title=f"KITTI Seq {sequence} — before/after optimization",
    )

    plot_ape_on_trajectory(
        gt_poses,
        optimized,
        title=f"KITTI-{sequence} {label} APE",
    )

    plot_metrics_over_time(
        gt_poses,
        {"KISS-ICP": kiss_poses, label: optimized},
        title=f"KITTI Seq {sequence}",
    )

    plot_graph(
        poses=optimized,
        loop_edges=graph.loop_edges,
        imu_edges=graph.imu_edges if graph.imu_edges else None,
        gt_poses=gt_poses,
        title=f"KITTI Seq {sequence} — Pose Graph",
    )


def run_milestone1(basedir: str, sequence: str):
    """Milestone 1: KISS-ICP + Scan Context loop closures + GTSAM pose graph."""

    dataset = KITTIDataset(basedir, sequence)
    n_frames = len(dataset)
    print(f"[M1] Sequence {sequence}: {n_frames} frames")

    odometry = LiDARodometry(max_range=100.0, min_range=5.0)
    graph = PoseGraph(NoiseSigmas())
    detector = LoopClosureDetector(
        sc_threshold=0.3,
        icp_fitness_threshold=0.6,
        icp_rmse_threshold=0.95,
        icp_max_distance=5.0,
        min_frame_gap=200,
    )
    loops: list = []

    it = iter(dataset)
    first_scan = next(it)
    pose = odometry.register(first_scan)
    graph.add_first_pose(pose)
    detector.add_frame(first_scan)

    pbar = tqdm(enumerate(it, start=1), total=n_frames - 1, desc="M1-GraphSLAM")
    for i, cloud in pbar:
        pose = odometry.register(cloud)
        delta = np.linalg.inv(odometry.poses[i - 1]) @ pose
        graph.add_odometry(pose, delta)

        loop = detector.add_frame(cloud)
        if loop is not None:
            graph.add_loop_closure(loop.idx_to, loop.idx_from, loop.T_relative)
            loops.append(loop)
            tqdm.write(
                f"  LOOP: {loop.idx_from} <-> {loop.idx_to}  "
                f"SC={loop.sc_dist:.3f}  fitness={loop.icp_fitness:.3f}  RMSE={loop.icp_rmse:.3f}"
            )
        pbar.set_postfix(loops=len(loops))

    print(f"\nTotal loop closures: {len(loops)}")
    optimized = graph.optimize()
    evaluate(sequence, dataset.gt_poses, odometry.poses, optimized, graph, "ISAM2 + loops")


def run_milestone2(basedir: str, raw_basedir: str, sequence: str):
    """Milestone 2: LiDAR-IMU Graph-SLAM (loosely-coupled).

    Uses M1's Pose3-only graph with two BetweenFactorPose3 per edge:
    one from KISS-ICP (tight) and one from IMU preintegration (loose).
    """

    dataset = KITTIDataset(basedir, sequence)
    imu_data = KITTIRawIMU(raw_basedir, sequence)
    n_frames = len(dataset)
    print(f"[M2] Sequence {sequence}: {n_frames} frames, IMU samples: {len(imu_data.timestamps)}")

    odometry = LiDARodometry(max_range=100.0, min_range=5.0)
    T_velo_imu = imu_data.T_velo_imu
    graph = PoseGraph(NoiseSigmas())
    detector = LoopClosureDetector(
        sc_threshold=0.3,
        icp_fitness_threshold=0.6,
        icp_rmse_threshold=0.95,
        icp_max_distance=5.0,
        min_frame_gap=200,
    )
    loops: list = []
    imu_factors = 0

    lidar_imu_times = imu_data.get_lidar_timestamps(n_frames)

    # First frame
    it = iter(dataset)
    first_scan = next(it)
    pose = odometry.register(first_scan)
    graph.add_first_pose(pose)
    detector.add_frame(first_scan)

    pbar = tqdm(enumerate(it, start=1), total=n_frames - 1, desc="M2-LIO-SLAM")
    for i, cloud in pbar:
        pose = odometry.register(cloud)
        delta = np.linalg.inv(odometry.poses[i - 1]) @ pose
        graph.add_odometry(pose, delta)

        # IMU gyro rotation as a loose second constraint
        t_start = lidar_imu_times[i - 1]
        t_end = lidar_imu_times[i]
        imu_meas = imu_data.get_imu_between(t_start, t_end)

        if len(imu_meas) >= 2:
            imu_rot = integrate_gyro(imu_meas, T_velo_imu)
            imu_delta = np.eye(4)
            imu_delta[:3, :3] = imu_rot  # from IMU gyroscope
            imu_delta[:3, 3] = delta[:3, 3]  # from KISS-ICP
            graph.add_imu_odometry(imu_delta)
            imu_factors += 1

        # Loop closure
        loop = detector.add_frame(cloud)
        if loop is not None:
            graph.add_loop_closure(loop.idx_to, loop.idx_from, loop.T_relative)
            loops.append(loop)
            tqdm.write(
                f"  LOOP: {loop.idx_from} <-> {loop.idx_to}  "
                f"SC={loop.sc_dist:.3f}  fitness={loop.icp_fitness:.3f}  RMSE={loop.icp_rmse:.3f}"
            )

        pbar.set_postfix(loops=len(loops), imu=imu_factors)

    print(f"\nTotal loop closures: {len(loops)}, IMU factors: {imu_factors}")
    optimized = graph.optimize()
    evaluate(sequence, dataset.gt_poses, odometry.poses, optimized, graph, "ISAM2+IMU+loops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI Graph-SLAM")
    parser.add_argument(
        "--milestone",
        type=int,
        choices=[1, 2],
        default=1,
        help="Milestone (1=LiDAR-only, 2=LiDAR-IMU)",
    )
    parser.add_argument("--basedir", default="data", help="Path to KITTI odometry data root")
    parser.add_argument(
        "--raw-basedir", default="data/raw", help="Path to KITTI raw data root (M2 only)"
    )
    parser.add_argument("--sequence", default="07", help="Sequence number (e.g. 07)")
    args = parser.parse_args()

    if args.milestone == 1:
        run_milestone1(args.basedir, args.sequence)
    else:
        run_milestone2(args.basedir, args.raw_basedir, args.sequence)
