# LIO-SAM Inspired Graph-SLAM on KITTI Benchmark

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Graph-SLAM on KITTI odometry benchmark, implemented in two milestones:

- **Milestone 1 (current):** LiDAR-only pipeline — KISS-ICP front-end, Scan Context loop closure detection with ICP verification, GTSAM ISAM2 pose graph optimization.
- **Milestone 2 (planned):** LIO-SAM inspired tightly-coupled LiDAR-IMU odometry, replacing the pure LiDAR front-end with IMU pre-integration to reduce drift and improve robustness.

## Architecture

```
LiDAR scan
    │
    ▼
KISS-ICP  ──────────────────────────► Odometry factor
    │                                        │
    ▼                                        ▼
Scan Context                          GTSAM ISAM2
 + KD-tree                           pose graph
    │                                        │
    ▼                                        ▼
ICP verification  ──────────────► Loop closure factor
```

**Front-end:** KISS-ICP registers consecutive scans using point-to-point ICP with a constant velocity motion model for initial pose prediction.

**Loop closure:** Each scan is encoded as a Scan Context descriptor — a polar 2D matrix of max point heights per (ring, sector) bin. Ring keys (per-ring mean vectors) are stored in a KD-tree for fast candidate retrieval. Candidates are verified with Open3D ICP using the Scan Context yaw estimate as the initial guess.

**Back-end:** GTSAM ISAM2 incrementally optimizes the pose graph as new odometry and loop closure factors arrive.

## Milestone 1 Results (LiDAR-only)

### Sequence 07 — short urban loop (0.7 km)

Seq 07 has a single loop closure between its start and end. KISS-ICP performs well on this short sequence (APE ~0.8 m), so loop closure has marginal numerical impact — but the system correctly detects the loop and the optimization is consistent.

| Method | APE rmse | APE mean | APE max |
|--------|----------|----------|---------|
| KISS-ICP | 0.799 m | 0.695 m | 1.430 m |
| ISAM2 + loops | 0.846 m | 0.733 m | 1.558 m |

RPE trans rmse: 0.025 m &nbsp;|&nbsp; RPE rot rmse: 0.079°

![Seq 07 trajectory](assets/kiss07.png)
![Seq 07 APE](assets/kiss07_APE.png)

<details>
<summary>Zoom views</summary>

![Seq 07 zoom](assets/kiss07_zoom.png)
![Seq 07 zoom 2](assets/kiss07_zoom2.png)
![Seq 07 APE zoom](assets/kiss07_APE_zoom.png)

</details>

---

### Sequence 06 — urban loop (1.2 km)

Loop closure reduces APE rmse from 2.79 m to 2.71 m (−3%). The improvement is visible in the trajectory: the optimized path closes tighter than raw odometry.

| Method | APE rmse | APE mean | APE max |
|--------|----------|----------|---------|
| KISS-ICP | 2.793 m | 2.247 m | 6.280 m |
| ISAM2 + loops | 2.708 m | 2.033 m | 5.718 m |

RPE trans rmse: 0.027 m &nbsp;|&nbsp; RPE rot rmse: 0.044°

![Seq 06 trajectory](assets/kiss06.png)
![Seq 06 APE](assets/kiss06_APE.png)

<details>
<summary>Zoom views</summary>

![Seq 06 zoom](assets/kiss06_zoom.png)
![Seq 06 APE zoom](assets/kiss06_APE_zoom.png)

</details>

---

### Sequence 00 — long urban sequence with multiple loops (3.7 km)

The largest sequence tested. Multiple loop closures are detected across the trajectory. APE rmse improves from 12.53 m to 10.91 m (−13%). The effect is clearly visible — the graph pulls sub-trajectories into alignment at each detected loop.

| Method | APE rmse | APE mean | APE max |
|--------|----------|----------|---------|
| KISS-ICP | 12.531 m | 11.177 m | 27.136 m |
| ISAM2 + loops | 10.910 m | 9.235 m | 25.765 m |

RPE trans rmse: 0.037 m &nbsp;|&nbsp; RPE rot rmse: 0.152°

![Seq 00 trajectory](assets/kiss00.png)
![Seq 00 APE](assets/kiss00_APE.png)

<details>
<summary>Zoom views</summary>

![Seq 00 zoom](assets/kiss00_zoom.png)
![Seq 00 zoom 2](assets/kiss00_zoom2.png)
![Seq 00 APE zoom](assets/kiss00_APE_zoom.png)

</details>

---

### Discussion

Loop closure benefit scales with sequence length and accumulated drift. On seq 07 (0.7 km, low drift), the single detected loop barely moves the numbers. On seq 00 (3.7 km, multiple loops), the improvement is clearly measurable. This is expected: graph optimization can only redistribute existing drift — the more drift there is to correct, the more visible the gain.

RPE is identical before/after optimization across all sequences. This confirms that loop closures affect global consistency (APE) but not local odometry accuracy (RPE), which is determined entirely by KISS-ICP.

The primary limitation at this stage is LiDAR-only operation — will be addressed in Milestone 2. IMU pre-integration (LIO-SAM style) will reduce front-end drift directly, making the baseline stronger and loop closures even more impactful, especially on longer sequences like seq 00.

## Setup

Tested on Linux x86_64:
- Ubuntu 24.04.4 LTS
- g++ 13.3.0
- Python 3.12
- System dependency: `libeigen3-dev`

```bash
sudo apt install libeigen3-dev
git submodule update --init
uv sync
scripts/build_scancontext.sh
uv run scripts/test_scancontext.py  # verify Scan Context works
```

## Usage

```bash
uv run main.py --sequence 00 --datadir data/
```

KITTI odometry data (sequences + calibration) should be placed under `data/`. Download the velodyne scans, ground truth poses, and calibration files from the [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The expected layout is:

```
data/
├── poses/
│   ├── 00.txt
│   ├── 01.txt
│   └── ...
└── sequences/
    ├── 00/
    │   ├── calib.txt
    │   ├── times.txt
    │   └── velodyne/
    │       ├── 000000.bin
    │       ├── 000001.bin
    │       └── ...
    ├── 01/
    └── ...
```

Note: KITTI distributes calibration files separately from the velodyne scans. Merge them into each sequence directory so that `calib.txt` sits alongside `velodyne/`.

## References

1. [KISS-ICP](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/vizzo2023ral.pdf) — Vizzo et al., RA-L 2023
2. [KITTI benchmark](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf) — Geiger et al., IJRR 2013
3. [LIO-SAM](https://arxiv.org/pdf/2007.00258) — Shan et al., IROS 2020
4. [Scan Context](https://ieeexplore.ieee.org/document/8593953) — Kim & Kim, IROS 2018
