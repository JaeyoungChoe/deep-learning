"""Render a rotating 3D point cloud GIF from Depth Anything 3 GLB output."""
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
import sys

EXPORT_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/sample_video_final"
GLB_PATH = os.path.join(EXPORT_DIR, "scene.glb")
OUT_GIF = os.path.join(EXPORT_DIR, "scene_3d.gif")
N_FRAMES = 36
FPS = 12

print(f"Loading {GLB_PATH}...")
scene = trimesh.load(GLB_PATH)

# Extract point cloud
pts, cols = None, None
for name, geom in scene.geometry.items():
    if isinstance(geom, trimesh.PointCloud):
        pts = np.array(geom.vertices)
        cols = np.array(geom.colors)[:, :3] / 255.0
        print(f"Point cloud: {pts.shape[0]} points")
        break

if pts is None:
    print("No point cloud found in GLB!")
    sys.exit(1)

# Subsample for faster rendering
MAX_RENDER_PTS = 200000
if pts.shape[0] > MAX_RENDER_PTS:
    idx = np.random.choice(pts.shape[0], MAX_RENDER_PTS, replace=False)
    pts, cols = pts[idx], cols[idx]
    print(f"Subsampled to {MAX_RENDER_PTS} points for rendering")

# Center and scale
center = np.median(pts, axis=0)
pts_c = pts - center
scale = np.percentile(np.abs(pts_c), 95)
if scale > 0:
    pts_c /= scale

# Render frames
print(f"Rendering {N_FRAMES} frames...")
frames = []
fig = plt.figure(figsize=(8, 6), dpi=80)

for i in range(N_FRAMES):
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")

    elev = 20 + 10 * np.sin(2 * np.pi * i / N_FRAMES)
    azim = 360 * i / N_FRAMES

    ax.scatter(
        pts_c[:, 0], pts_c[:, 2], pts_c[:, 1],
        c=cols, s=0.3, alpha=0.8, edgecolors="none",
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    frames.append(buf)

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{N_FRAMES}")

plt.close(fig)

print(f"Saving GIF to {OUT_GIF}...")
imageio.mimsave(OUT_GIF, frames, fps=FPS, loop=0)

file_size = os.path.getsize(OUT_GIF) / 1024 / 1024
print(f"Done! {OUT_GIF} ({file_size:.1f} MB)")
