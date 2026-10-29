"""Compare a reconstructed STL with its CAD reference (Chamfer distance, F-score).

    python eval/chamfer_eval.py <reconstruction.stl> <reference.stl> [--parts 0,1,3,4] [--out results/x]

The reconstruction has arbitrary scale, rotation and position (COLMAP has no metric scale), so it is first
aligned to the reference with a similarity transform: 24 axis-aligned starts from the principal axes,
each refined by ICP with scale (Umeyama); the start with the lowest RMS wins. Distances are then
reported in the reference's units (mm for printable STLs). Because scale is fitted, the absolute size of
the reconstruction is NOT evaluated - only its shape.

--parts selects connected components of the reference (largest first, as printed by --list), e.g. to
compare only the dragon of a two-colour print when the segmentation picked only that colour.
"""
import argparse
import itertools
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

N_POINTS = 60000


def load(path, parts=None):
    m = trimesh.load(path, force="mesh")
    if parts is not None:
        comps = sorted(m.split(only_watertight=False), key=lambda p: -len(p.faces))
        m = trimesh.util.concatenate([comps[i] for i in parts])
    return m


def sample(m, n=N_POINTS, seed=0):
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


def umeyama(src, dst):
    """Similarity transform (s, R, t) minimizing |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def icp(src, dst_tree, dst, s, R, t, iters=60, trim=0.9):
    """Symmetric trimmed ICP with scale.

    Matches go both ways (src->dst and dst->src). One-way matching lets the fitted scale collapse:
    a shrunken reconstruction glued onto one part of the reference has small src->dst distances.
    The worst 10 % of matches in each direction are ignored (floaters, missing parts).
    """
    for _ in range(iters):
        cur = s * src @ R.T + t
        d1, i1 = dst_tree.query(cur)
        d2, i2 = cKDTree(cur).query(dst)
        k1, k2 = d1 <= np.quantile(d1, trim), d2 <= np.quantile(d2, trim)
        s, R, t = umeyama(np.vstack([src[k1], src[i2[k2]]]), np.vstack([dst[i1[k1]], dst[k2]]))
    cur = s * src @ R.T + t
    d1, _ = dst_tree.query(cur)
    d2, _ = cKDTree(cur).query(dst)
    d = np.concatenate([np.sort(d1)[: int(trim * len(d1))], np.sort(d2)[: int(trim * len(d2))]])
    return s, R, t, float(np.sqrt(np.mean(d ** 2)))


def principal_frame(p):
    c = p.mean(0)
    _, _, Vt = np.linalg.svd(p - c, full_matrices=False)
    return c, Vt  # rows = principal axes


def align(rec, ref):
    ref_full_points = ref
    c_rec, A = principal_frame(rec)
    c_ref, B = principal_frame(ref)
    s0 = np.sqrt(((ref - c_ref) ** 2).sum(1).mean() / ((rec - c_rec) ** 2).sum(1).mean())
    sub = rec[np.random.default_rng(0).choice(len(rec), 8000, replace=False)]
    ref_sub = ref[np.random.default_rng(1).choice(len(ref), 8000, replace=False)]
    ref_tree, ref = cKDTree(ref_sub), ref_sub  # coarse search on subsets, final ICP on all points below
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            P = np.zeros((3, 3))
            for i, j in enumerate(perm):
                P[i, j] = signs[i]
            R0 = B.T @ P @ A  # rec principal axes -> (permuted, flipped) ref principal axes
            if np.linalg.det(R0) < 0:
                continue
            t0 = c_ref - s0 * R0 @ c_rec
            s, R, t, rms = icp(sub, ref_tree, ref, s0, R0, t0, iters=30)
            if best is None or rms < best[3]:
                best = (s, R, t, rms)
    ref_full = ref_full_points
    s, R, t, rms = icp(rec, cKDTree(ref_full), ref_full, *best[:3], iters=40)
    return s, R, t, rms


def metrics(rec_aligned, ref, size):
    d_acc, _ = cKDTree(ref).query(rec_aligned)   # reconstruction -> reference (accuracy)
    d_com, _ = cKDTree(rec_aligned).query(ref)   # reference -> reconstruction (completeness)
    out = {
        "chamfer_mm": float((d_acc.mean() + d_com.mean()) / 2),
        "accuracy_mean_mm": float(d_acc.mean()), "accuracy_median_mm": float(np.median(d_acc)),
        "completeness_mean_mm": float(d_com.mean()), "completeness_median_mm": float(np.median(d_com)),
        "reference_diagonal_mm": float(size),
    }
    out["chamfer_pct_of_diagonal"] = 100 * out["chamfer_mm"] / size
    for tau in (1.0, 2.0, 5.0):
        p, r = (d_acc < tau).mean(), (d_com < tau).mean()
        out[f"fscore_{tau:g}mm"] = float(2 * p * r / (p + r)) if p + r else 0.0
    return out, d_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recon")
    ap.add_argument("reference")
    ap.add_argument("--parts", help="comma-separated component indices of the reference")
    ap.add_argument("--list", action="store_true", help="list reference components and exit")
    ap.add_argument("--out", default=None, help="output prefix for aligned STL + metrics JSON")
    a = ap.parse_args()

    if a.list:
        m = trimesh.load(a.reference, force="mesh")
        for i, p in enumerate(sorted(m.split(only_watertight=False), key=lambda p: -len(p.faces))):
            print(i, len(p.faces), "tris, extents", p.extents.round(1))
        return

    parts = [int(x) for x in a.parts.split(",")] if a.parts else None
    ref_mesh, rec_mesh = load(a.reference, parts), load(a.recon)
    ref, rec = sample(ref_mesh), sample(rec_mesh, seed=1)
    s, R, t, rms = align(rec, ref)
    rec_aligned = s * rec @ R.T + t
    out, _ = metrics(rec_aligned, ref, float(np.linalg.norm(ref_mesh.extents)))
    out.update(icp_trimmed_rms_mm=rms, fitted_scale=float(s), reference_parts=parts,
               reference_extents_mm=ref_mesh.extents.round(2).tolist())
    print(json.dumps(out, indent=2))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        aligned = rec_mesh.copy()
        aligned.apply_transform(np.vstack([np.hstack([s * R, t[:, None]]), [0, 0, 0, 1]]))
        aligned.export(a.out + "_aligned.stl")
        ref_mesh.export(a.out + "_reference.stl")
        json.dump(out, open(a.out + "_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
