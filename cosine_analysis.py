"""
cosine_analysis.py — Measure redundancy among BLIP-2's 32 Q-Former query tokens.

The Q-Former compresses the whole image into 32 query vectors (768-dim each).
Question: are all 32 doing distinct work, or are many of them near-duplicates?

If two query tokens point in nearly the same direction (cosine similarity ~1),
they carry nearly the same information — so the effective bottleneck is even
tighter than 32. This script:

  1. Loads the saved Q-Former tensor(s) from inspect_qformer.py --save-tensors
  2. Computes the 32x32 pairwise cosine similarity matrix
  3. Reports summary numbers (mean off-diagonal similarity, effective rank)
  4. Saves a heatmap figure

Usage:
    python cosine_analysis.py --tensor results/CLEVR_val_000000_qformer.npy
    python cosine_analysis.py --all
"""

import argparse
import glob
import os

# ── Windows LAPACK/OpenMP fix ────────────────────────────────────────
# Set before numpy import. Helps with MKL/OpenMP threading conflicts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from config import RESULTS_DIR


def cosine_matrix(tokens):
    """Pairwise cosine similarity matrix. tokens: (num_tokens, dim) -> (num_tokens, num_tokens)."""
    tokens = np.squeeze(np.asarray(tokens, dtype=np.float64))
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got {tokens.shape}")

    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    unit = tokens / norms

    return unit @ unit.T


def off_diagonal_mean(matrix):
    """Average cosine similarity between DIFFERENT tokens (ignores diagonal)."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return matrix[mask].mean()


def effective_rank(tokens, threshold=0.99):
    """
    How many principal components explain `threshold` (99%) of the variance.
    Uses SVD: the singular values squared are the variances along each
    principal direction.
    """
    tokens = np.squeeze(np.asarray(tokens, dtype=np.float64))
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D array, got {tokens.shape}")

    # Center the tokens (variance is spread around the mean)
    centered = tokens - tokens.mean(axis=0, keepdims=True)

    # SVD: singular values s; variance along each direction = s^2
    s = np.linalg.svd(centered, compute_uv=False)
    var = s ** 2

    total = var.sum()
    if total == 0 or not np.isfinite(total):
        return 1

    var_ratio = var / total
    cumulative = np.cumsum(var_ratio)

    return int(np.searchsorted(cumulative, threshold) + 1)


def load_qformer_tensor(path):
    """Load a saved Q-Former tensor and squeeze to (num_tokens, dim)."""
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float64)
    n_bad = np.count_nonzero(~np.isfinite(arr))
    if n_bad:
        print(f"  ⚠ {n_bad} non-finite value(s) (inf/nan) found — replacing with 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def analyze_one(path, save_fig=True):
    """Analyze a single Q-Former tensor. Returns (matrix, mean_sim, eff_rank)."""
    tokens = load_qformer_tensor(path)
    print(f"  loaded tensor shape: {tokens.shape}")
    n_tokens, dim = tokens.shape

    matrix = cosine_matrix(tokens)
    print(matrix)
    mean_sim = off_diagonal_mean(matrix)
    eff_rank = effective_rank(tokens)

    name = os.path.basename(path).replace("_qformer.npy", "")
    print(f"\n── {name} ──")
    print(f"  Query tokens:         {n_tokens} × {dim}")
    print(f"  Mean off-diag cosine: {mean_sim:.3f}   (0 = all distinct, 1 = all identical)")
    print(f"  Effective rank (99%): {eff_rank}/{n_tokens}   "
          f"({n_tokens - eff_rank} tokens are ~redundant)")

    if save_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(f"Q-Former token cosine similarity\n{name}", fontsize=11)
        ax.set_xlabel("Query token index")
        ax.set_ylabel("Query token index")
        plt.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
        out_path = os.path.join(RESULTS_DIR, f"{name}_cosine.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap → {out_path}")

    return matrix, mean_sim, eff_rank


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity of Q-Former query tokens")
    parser.add_argument("--tensor", help="Path to one *_qformer.npy file")
    parser.add_argument("--all", action="store_true",
                        help="Analyze every *_qformer.npy in the results folder")
    args = parser.parse_args()

    if args.all:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_qformer.npy")))
        if not paths:
            print(f"No *_qformer.npy files found in {RESULTS_DIR}")
            print("Run inspect_qformer.py --save-tensors first.")
            return

        print(f"Found {len(paths)} Q-Former tensors\n")
        all_sims = []
        all_ranks = []
        for p in paths:
            _, mean_sim, eff_rank = analyze_one(p, save_fig=True)
            all_sims.append(mean_sim)
            all_ranks.append(eff_rank)

        print(f"\n{'='*50}")
        print("  AGGREGATE across all images")
        print(f"{'='*50}")
        print(f"  Mean off-diag cosine: {np.mean(all_sims):.3f} ± {np.std(all_sims):.3f}")
        print(f"  Mean effective rank:  {np.mean(all_ranks):.1f}/32")
        print(f"{'='*50}\n")

    elif args.tensor:
        analyze_one(args.tensor, save_fig=True)

    else:
        print("Provide --tensor <path> or --all")


if __name__ == "__main__":
    main()