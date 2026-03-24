"""Milestone 1: KISS-ICP + Scan Context loop closures + GTSAM pose graph.

Usage:
    uv run milestone_1.py --sequence 07
    uv run milestone_1.py --sequence 07 --basedir /path/to/kitti
"""

import argparse

import numpy as np
from tqdm import tqdm

from source.eval import (
    compute_ape,
    compute_rpe,
    compute_rpe_rotation,
    plot_ape_on_trajectory,
    plot_trajectories,
    print_metrics,
)
from source.graph import NoiseSigmas, PoseGraph
from source.kiss import LiDARodometry
from source.kitti import KITTIDataset
from source.loop_closure import LoopClosureDetector


def main(basedir: str, sequence: str):
    # Loading kitti sequence scans
    dataset = KITTIDataset(basedir, sequence)
    n_frames = len(dataset)
    print(f"Sequence {sequence}: {n_frames} frames")

    # Odometry + loop detection + GTSAM graph
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

    pbar = tqdm(enumerate(it, start=1), total=n_frames - 1, desc="KITTI-GraphSLAM")
    for i, cloud in pbar:
        pose = odometry.register(cloud)
        delta = np.linalg.inv(odometry.poses[i - 1]) @ pose
        graph.add_odometry(pose, delta)

        loop = detector.add_frame(cloud)
        if loop is not None:
            graph.add_loop_closure(loop.idx_to, loop.idx_from, loop.T_relative)
            loops.append(loop)
            tqdm.write(f"  LOOP: {loop.idx_from} <-> {loop.idx_to}  "
                        f"SC={loop.sc_dist:.3f}  fitness={loop.icp_fitness:.3f}  RMSE={loop.icp_rmse:.3f}")

        pbar.set_postfix(loops=len(loops))

    print(f"\nTotal loop closures: {len(loops)}")

    # Graph optimization 
    optimized = graph.optimize()

    # Metrics
    gt = dataset.gt_poses
    kiss_ape, _ = compute_ape(gt, odometry.poses)
    kiss_rpe = compute_rpe(gt, odometry.poses)
    kiss_rpe_rot = compute_rpe_rotation(gt, odometry.poses)

    opt_ape, _ = compute_ape(gt, optimized)
    opt_rpe = compute_rpe(gt, optimized)
    opt_rpe_rot = compute_rpe_rotation(gt, optimized)

    print(f"\nSequence {sequence} — results:")
    print_metrics("KISS-ICP (raw)", kiss_ape, kiss_rpe, kiss_rpe_rot)
    print_metrics("ISAM2 + loop closures", opt_ape, opt_rpe, opt_rpe_rot)

    # Plots
    plot_trajectories(
        trajectories={"KISS-ICP": odometry.poses, "ISAM2 + loops": optimized},
        gt_poses=gt,
        title=f"KITTI Seq {sequence} — before/after loop closure optimization",
    )

    plot_ape_on_trajectory(
        gt, optimized,
        title=f"KITTI-{sequence} + loop closures APE",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="data", help="Path to KITTI odometry data root")
    parser.add_argument("--sequence", default="07", help="Sequence number (e.g. 07)")
    args = parser.parse_args()
    main(args.basedir, args.sequence)
