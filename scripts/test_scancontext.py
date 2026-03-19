"""
Smoke test for pyscancontext bindings.

Simulates a vehicle driving in a circle (60 frames), so the last frame
revisits the start. Expects detect_loop() to fire and report a match
with distance < 0.2 once enough history is accumulated (NUM_EXCLUDE_RECENT=50).
"""
import numpy as np
import pyscancontext as sc


def make_ring_scan(center_xy, n_points=2000, noise=0.05, n_rings=16):
    """
    Fake a 16-ring LiDAR scan centred at `center_xy`.
    Points lie on concentric cylinders at different elevations.
    """
    pts = []
    for ring in range(n_rings):
        elev = np.deg2rad(-15 + ring * 2)        # -15 to +15 deg
        r = 10.0 + np.random.randn(n_points) * noise
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = center_xy[0] + r * np.cos(angles)
        y = center_xy[1] + r * np.sin(angles)
        z = np.full(n_points, r[0] * np.tan(elev))
        pts.append(np.stack([x, y, z], axis=1))
    return np.concatenate(pts, axis=0)


def main():
    scm = sc.SCManager()
    scm.print_parameters()

    LOOP_THRES = 0.2
    N_FRAMES = 65          # >50 so detect_loop can fire; last frame revisits frame 0
    RADIUS = 30.0

    print(f"\nGenerating {N_FRAMES} frames along a circular path (r={RADIUS}m)...")
    angles = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)
    centers = [(RADIUS * np.cos(a), RADIUS * np.sin(a)) for a in angles]

    loop_found = False
    for i, center in enumerate(centers):
        cloud = make_ring_scan(center)
        scd = scm.make_scancontext(cloud)
        scm.add_scancontext(scd)

        nn_idx, nn_dist, yaw_diff = scm.detect_loop()

        if nn_idx == -1:
            print(f"  frame {i:3d}: accumulating history...")
            continue

        marker = " <-- LOOP" if nn_dist < LOOP_THRES else ""
        print(f"  frame {i:3d}: nn={nn_idx:3d}  dist={nn_dist:.3f}  yaw={np.rad2deg(yaw_diff):+.1f} deg{marker}")

        if nn_dist < LOOP_THRES:
            loop_found = True

    print()
    if loop_found:
        print("PASS — pyscancontext detected a loop closure correctly.")
    else:
        print("FAIL — no loop detected (dist never dropped below threshold).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
