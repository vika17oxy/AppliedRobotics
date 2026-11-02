"""Trained splatfacto model + SAM2 masks -> watertight STL.

Runs inside the nerfstudio venv (/content/nsenv), not the Colab kernel:
    /content/nsenv/bin/python hull.py <config.yml> <out.stl> [voxels]

Writes <out.stl> (silhouettes + 3DGS depth) and <out>_hull.stl (silhouettes only, for comparison).

Why not mesh the Gaussians directly: masked training does not delete background Gaussians from the
COLMAP initialization, and thin/dark parts get too few Gaussians for Poisson. Why not silhouettes only:
with ~20-40 views the visual hull is a faceted polyhedron (flat cuts, wedges) and cannot show concavities.

So both are combined in one voxel grid around the object:
  1. silhouettes: a voxel survives if it projects inside the mask in >= 95 % of the views that see it;
  2. 3DGS depth: every training view is rendered from the trained scene; a voxel is free space if, in at
     least DEPTH_VOTES views, it lies inside the mask but clearly in front of the rendered surface
     (the camera sees past it). This removes the wedges of the hull and restores dents.
The occupancy is blurred slightly before marching cubes, which removes the voxel staircase.
Cameras, Gaussians and voxels come from the same loaded pipeline, so they share one coordinate frame.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # before torch: inductor needs libcuda.so, missing on Colab

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter, label
from skimage.measure import marching_cubes
from nerfstudio.utils.eval_utils import eval_setup

args = [a for a in sys.argv[1:] if not a.startswith("--")]
cfg, out = args[0], args[1]
N = int(args[2]) if len(args) > 2 else 288
MIN_ACCUM = 0.5        # only trust rendered depth where the scene is opaque enough
# depth carving: remove a voxel if at least FRACTION of the views with trusted depth there see past it
# (and at least 2 views), by more than MARGIN voxels. Textureless parts give noisy depth, so a plain
# "any 2 views" rule shredded the black plate of the dragon test. (fraction, margin) per variant:
VARIANTS = {"": (0.8, 6.0)}                       # default output (best on the dragon test)
if "--variants" in sys.argv:                      # also write a sweep, to pick settings per capture
    VARIANTS.update({"_soft": (0.6, 5.0), "_mid": (0.4, 4.0)})

_, pipe, _, _ = eval_setup(Path(cfg), test_mode="test")
model, ds = pipe.model, pipe.datamanager.train_dataset
cams = ds.cameras
c2w = cams.camera_to_worlds.cuda()
fx, fy, cx, cy = [getattr(cams, a).cuda().ravel() for a in ("fx", "fy", "cx", "cy")]
masks = [torch.from_numpy(np.array(Image.open(f)) > 127).cuda() for f in ds._dataparser_outputs.mask_filenames]
# a point must be seen in this many views to be judged at all; 10 for normal captures, fewer for 5-19 photos
MIN_SEEN = max(3, min(10, len(masks) // 2))


def project(P, k):
    """Pixel coordinates, camera-space depth and visibility of points P (n,3) in view k."""
    H, W = masks[k].shape[:2]
    R, t = c2w[k][:, :3], c2w[k][:, 3]
    Pc = (P - t) @ R  # OpenGL camera: z points backwards
    z = -Pc[:, 2]
    u = fx[k] * Pc[:, 0] / z + cx[k]
    v = -fy[k] * Pc[:, 1] / z + cy[k]
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, ok


def mask_votes(P):
    """For points P (n,3): in how many views they are visible, and in how many inside the mask."""
    inside = torch.zeros(len(P), device="cuda")
    seen = torch.zeros(len(P), device="cuda")
    for k, M in enumerate(masks):
        u, v, _, ok = project(P, k)
        seen += ok
        inside[ok] += M[v[ok].long(), u[ok].long()].float()
    return inside, seen


def render_depth(k):
    """Rendered depth (H,W) of view k and where it can be trusted (opaque and inside the mask)."""
    with torch.no_grad():
        o = model.get_outputs_for_camera(cams[k:k + 1].to("cuda"))
    H, W = masks[k].shape[:2]
    depth = o["depth"][..., 0]
    acc = o.get("accumulation", torch.ones_like(o["depth"]))[..., 0]
    if depth.shape != (H, W):  # renders follow the camera size; masks should match, but be safe
        depth = F.interpolate(depth[None, None], (H, W), mode="nearest")[0, 0]
        acc = F.interpolate(acc[None, None], (H, W), mode="nearest")[0, 0]
    return depth, (acc > MIN_ACCUM) & masks[k]


def to_mesh(field, lo, step, path):
    verts, faces, _, _ = marching_cubes(np.pad(field, 1), 0.5, spacing=tuple(step))
    m = trimesh.Trimesh(verts + lo - step, faces)
    m = max(m.split(only_watertight=False), key=lambda c: len(c.faces))
    trimesh.smoothing.filter_taubin(m, iterations=10)
    m.fix_normals()
    m.export(path)
    print(Path(path).name, "|", len(m.vertices), "verts |", len(m.faces), "tris | box:", m.extents.round(3),
          "| watertight:", m.is_watertight)
    return m


# 1) bounding box of the object: Gaussians that fall inside the mask in >=90 % of views
X = model.means.detach()
inside, seen = mask_votes(X)
obj = X[(inside / seen.clamp(min=1) >= 0.9) & (seen >= MIN_SEEN)].cpu().numpy()
if len(obj) < 100:
    sys.exit(f"Only {len(obj)} Gaussians inside the masks - segmentation or poses failed")
lo, hi = np.percentile(obj, 0.5, axis=0), np.percentile(obj, 99.5, axis=0)
pad = 0.15 * (hi - lo)
lo, hi = lo - pad, hi + pad
step = (hi - lo) / (N - 1)
print("Gaussians:", len(X), "| object:", len(obj), "| voxel:", step.round(5))


def fit_floor():
    """Plane the object stands on, from background Gaussians around it (RANSAC), oriented towards the cameras.

    No camera sees the underside of an object lying on a surface, so silhouettes and depth both leave a thick
    bulge below it. The supporting surface is usually well textured and full of Gaussians, so its plane can be
    fitted and everything below it cut off. Returns (normal, offset) or None if no trustworthy plane is found.
    """
    bg = X[(inside / seen.clamp(min=1) <= 0.05) & (seen >= MIN_SEEN)].cpu().numpy()
    centre, radius = obj.mean(0), np.linalg.norm(hi - lo) / 2
    bg = bg[np.linalg.norm(bg - centre, axis=1) < 2.0 * radius]
    if len(bg) < 200:
        print("floor: too few background Gaussians near the object - not clipped")
        return None
    rng, tol = np.random.default_rng(0), 0.02 * radius
    best, best_in = None, 0
    for _ in range(1000):
        a, b, c = bg[rng.choice(len(bg), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        cnt = (np.abs((bg - a) @ n) < tol).sum()
        if cnt > best_in:
            best, best_in = (n, a), cnt
    n, a = best
    pts = bg[np.abs((bg - a) @ n) < tol]
    c0 = pts.mean(0)
    n = np.linalg.svd(pts - c0)[2][-1]                       # least-squares refit on the inliers
    cams_c = c2w[:, :, 3].cpu().numpy()
    if np.mean((cams_c - c0) @ n) < 0:                       # normal points to where the cameras are
        n = -n
    below = np.mean((obj - c0) @ n < -tol)
    print(f"floor: {len(pts)}/{len(bg)} background Gaussians on the plane, {100 * below:.1f} % of the object below it")
    if best_in < 0.2 * len(bg) or below > 0.15:
        print("floor: plane not trustworthy - not clipped")
        return None
    return n, float(n @ c0)


inside, seen = mask_votes(X)  # (again for the background test in fit_floor)
floor = fit_floor()
del inside, seen

# 2) silhouettes (and nothing below the floor)
ax = [torch.linspace(float(lo[i]), float(hi[i]), N, device="cuda") for i in range(3)]
G = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
inside, seen = mask_votes(G)
hull = (inside / seen.clamp(min=1) >= 0.95) & (seen >= MIN_SEEN)
del inside, seen
print("silhouette voxels:", int(hull.sum()))
if floor is not None:
    n_f, d_f = floor
    hull &= (G @ torch.tensor(n_f, dtype=G.dtype, device="cuda")) > d_f
    print("above the floor:", int(hull.sum()))

# 3) free space from rendered 3DGS depth, only for voxels the silhouettes kept
idx = torch.nonzero(hull).ravel()
P = G[idx]
margins = sorted({m for _, m in VARIANTS.values()})
free = {m: torch.zeros(len(P), device="cuda") for m in margins}
trusted = torch.zeros(len(P), device="cuda")
for k in range(len(masks)):
    depth, trust = render_depth(k)
    u, v, z, ok = project(P, k)
    sel = torch.nonzero(ok).ravel()
    ui, vi = u[ok].long(), v[ok].long()
    t = trust[vi, ui]
    trusted[sel[t]] += 1
    for m in margins:
        free[m][sel[t & (z[ok] < depth[vi, ui] - m * float(step.mean()))]] += 1


def clean(occ):
    """One solid piece: no inner voids, no thin spikes, only the largest connected component."""
    occ = occ.reshape(N, N, N).cpu().numpy()
    occ = binary_opening(occ, iterations=1)
    lab, n = label(occ)
    if n > 1:
        occ = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return binary_fill_holes(occ)


# 4) surfaces: a light blur of the 0/1 field removes the voxel staircase before marching cubes
def field(occ):
    return gaussian_filter(occ.astype(np.float32), sigma=1.0)


to_mesh(field(clean(hull)), lo, step, out[:-4] + "_hull.stl")
for suffix, (frac, m) in VARIANTS.items():
    carved = hull.clone()
    gone = (free[m] >= 2) & (free[m] >= frac * trusted)
    carved[idx[gone]] = False
    print(f"depth carving{suffix or ' (default)'}: fraction {frac}, margin {m} voxels ->",
          f"{100 * gone.float().mean().item():.1f} % of the silhouette volume removed")
    to_mesh(field(clean(carved)), lo, step, out[:-4] + suffix + ".stl")
