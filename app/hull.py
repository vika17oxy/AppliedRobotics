"""Trained splatfacto model + SAM2 masks -> watertight STL via visual hull.

Runs inside the nerfstudio venv (/content/nsenv), not the Colab kernel:
    /content/nsenv/bin/python hull.py <config.yml> <out.stl> [voxels]

Gaussians alone give a ragged mesh on dark/thin parts (few Gaussians, Poisson
cuts low-density regions). The masks have clean silhouettes everywhere, so the
object is reconstructed as the set of voxels that project inside the mask in
(almost) every view where they are visible. Camera poses come from the same
loaded pipeline, so voxels, Gaussians and cameras share one coordinate frame.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from skimage.measure import marching_cubes
from nerfstudio.utils.eval_utils import eval_setup

cfg, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 256

_, pipe, _, _ = eval_setup(Path(cfg), test_mode="test")
ds = pipe.datamanager.train_dataset
cams = ds.cameras
c2w = cams.camera_to_worlds.cuda()
fx, fy, cx, cy = [getattr(cams, a).cuda().ravel() for a in ("fx", "fy", "cx", "cy")]
masks = [torch.from_numpy(np.array(Image.open(f)) > 127).cuda() for f in ds._dataparser_outputs.mask_filenames]


def mask_votes(P):
    """For points P (n,3): in how many views they are visible, and in how many inside the mask."""
    inside = torch.zeros(len(P), device="cuda")
    seen = torch.zeros(len(P), device="cuda")
    for k, M in enumerate(masks):
        H, W = M.shape[:2]
        R, t = c2w[k][:, :3], c2w[k][:, 3]
        Pc = (P - t) @ R  # OpenGL camera: z points backwards
        z = -Pc[:, 2]
        u = fx[k] * Pc[:, 0] / z + cx[k]
        v = -fy[k] * Pc[:, 1] / z + cy[k]
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        seen += ok
        inside[ok] += M[v[ok].long(), u[ok].long()].float()
    return inside, seen


# 1) bounding box of the object: Gaussians that fall inside the mask in >=90 % of views
X = pipe.model.means.detach()
inside, seen = mask_votes(X)
obj = X[(inside / seen.clamp(min=1) >= 0.9) & (seen >= 10)].cpu().numpy()
if len(obj) < 100:
    sys.exit(f"Only {len(obj)} Gaussians inside the masks - segmentation or poses failed")
lo, hi = np.percentile(obj, 0.5, axis=0), np.percentile(obj, 99.5, axis=0)
pad = 0.15 * (hi - lo)
lo, hi = lo - pad, hi + pad
print("Gaussians:", len(X), "| object:", len(obj))

# 2) voxel carving inside that box
ax = [torch.linspace(float(lo[i]), float(hi[i]), N, device="cuda") for i in range(3)]
G = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
inside, seen = mask_votes(G)
occ = ((inside / seen.clamp(min=1) >= 0.95) & (seen >= 10)).reshape(N, N, N).cpu().numpy()
print("occupied voxels:", int(occ.sum()))

# 3) surface, keep the largest piece, smooth voxel steps away
step = (hi - lo) / (N - 1)
verts, faces, _, _ = marching_cubes(np.pad(occ.astype(np.float32), 1), 0.5, spacing=tuple(step))
m = trimesh.Trimesh(verts + lo - step, faces)
m = max(m.split(only_watertight=False), key=lambda c: len(c.faces))
trimesh.smoothing.filter_taubin(m, iterations=20)
m.fix_normals()
m.export(out)
print("STL:", len(m.vertices), "verts |", len(m.faces), "tris | box:", m.extents.round(3), "| watertight:", m.is_watertight)
