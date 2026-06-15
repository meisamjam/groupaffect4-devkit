#!/usr/bin/env python3
"""
Refine a 3D skeleton by filtering on multi-view confidence, removing
outliers, interpolating short gaps, and temporal smoothing.

Pipeline stages
---------------
1. **Quality gate** — reject joints below thresholds:
   - ``min_confidence``  (2D detector confidence, aggregated across views)
   - ``max_reproj_px``   (mean reprojection error after triangulation)
   - ``min_cameras``     (number of cameras that contributed)

2. **Velocity outlier rejection** — flag joints whose frame-to-frame
   velocity exceeds ``max_velocity_mm`` (Euclidean jump between
   consecutive valid frames).  Isolated spikes are NaN-ed out.

3. **Gap interpolation** — fill NaN gaps up to ``max_gap_frames`` using
   cubic spline (≥4 surrounding valid points) or linear interpolation.

4. **Temporal smoothing (boosting)** — 2nd-order Butterworth low-pass
   filter, applied per-joint per-coordinate.  Cutoff frequency set by
   ``smooth_cutoff_hz`` relative to the skeleton's frame rate.

The tool writes a new ``.npy`` file (with ``_refined`` suffix by default)
and a companion ``_refined.json`` with applied parameters and QC stats.

Usage
-----
    # Default refinement
    python tools/refine_skeleton_3d.py \\
        --input new_data/ses-20260202_test/skeleton_3d_mediapipe.npy

    # Tighter quality gate + aggressive smoothing
    python tools/refine_skeleton_3d.py \\
        --input new_data/ses-20260202_test/skeleton_3d_mediapipe.npy \\
        --min-confidence 0.5 \\
        --max-reproj 15 \\
        --min-cameras 3 \\
        --smooth-cutoff 3.0

    # Upper-body only
    python tools/refine_skeleton_3d.py \\
        --input new_data/ses-20260202_test/skeleton_3d_mediapipe.npy \\
        --upper-body

Dependencies: numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Upper-body keypoint indices (BODY_25)
UPPER_BODY_KP = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18}

# Column indices in the 7-dim skeleton
IX, IY, IZ = 0, 1, 2
ICONF = 3
IREPROJ = 4
INCAMS = 5
IGROUP = 6


# ---------------------------------------------------------------------------
# 1. Quality gate
# ---------------------------------------------------------------------------

def quality_gate(
    data: np.ndarray,
    min_confidence: float = 0.3,
    max_reproj_px: float = 20.0,
    min_cameras: int = 2,
    upper_body: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    NaN-out joints that fail quality thresholds.

    Args:
        data: (F, P, K, 7) skeleton array (modified in-place copy)
        min_confidence: minimum aggregated 2D confidence
        max_reproj_px: maximum allowed reprojection error
        min_cameras: minimum number of cameras used
        upper_body: if True, also NaN-out non-upper-body keypoints

    Returns:
        filtered data, stats dict
    """
    F, P, K, _ = data.shape
    before_valid = int(np.sum(~np.isnan(data[:, :, :, IX])))

    n_conf_reject = 0
    n_reproj_reject = 0
    n_cams_reject = 0
    n_body_reject = 0

    for pi in range(P):
        for ki in range(K):
            if upper_body and ki not in UPPER_BODY_KP:
                mask = ~np.isnan(data[:, pi, ki, IX])
                n_body_reject += int(mask.sum())
                data[:, pi, ki, :3] = np.nan
                continue

            for fi in range(F):
                if np.isnan(data[fi, pi, ki, IX]):
                    continue

                conf = data[fi, pi, ki, ICONF]
                reproj = data[fi, pi, ki, IREPROJ]
                ncams = data[fi, pi, ki, INCAMS]

                reject = False
                if conf < min_confidence:
                    n_conf_reject += 1
                    reject = True
                if reproj >= 0 and reproj > max_reproj_px:
                    n_reproj_reject += 1
                    reject = True
                if ncams < min_cameras:
                    n_cams_reject += 1
                    reject = True

                if reject:
                    data[fi, pi, ki, :3] = np.nan

    after_valid = int(np.sum(~np.isnan(data[:, :, :, IX])))

    stats = {
        "before_valid": before_valid,
        "after_valid": after_valid,
        "removed_total": before_valid - after_valid,
        "removed_low_confidence": n_conf_reject,
        "removed_high_reproj": n_reproj_reject,
        "removed_few_cameras": n_cams_reject,
        "removed_lower_body": n_body_reject,
    }
    return data, stats


# ---------------------------------------------------------------------------
# 2. Velocity outlier rejection
# ---------------------------------------------------------------------------

def velocity_outlier_rejection(
    data: np.ndarray,
    max_velocity_mm: float = 300.0,
    fps: float = 30.0,
) -> tuple[np.ndarray, int]:
    """
    NaN-out joints with implausible frame-to-frame velocity.

    Velocity is computed as Euclidean distance between consecutive valid
    frames, normalized to mm/frame (assuming data is in mm).

    Returns:
        filtered data, count of rejected joints
    """
    F, P, K, _ = data.shape
    n_rejected = 0

    for pi in range(P):
        for ki in range(K):
            coords = data[:, pi, ki, :3]  # (F, 3)
            valid_mask = ~np.isnan(coords[:, 0])
            valid_idx = np.where(valid_mask)[0]

            if len(valid_idx) < 2:
                continue

            # Compute velocities between consecutive valid frames
            for i in range(len(valid_idx) - 1):
                f0 = valid_idx[i]
                f1 = valid_idx[i + 1]
                dt_frames = f1 - f0
                dist = np.linalg.norm(coords[f1] - coords[f0])
                velocity = dist / dt_frames  # mm per frame

                if velocity > max_velocity_mm:
                    # NaN-out the later point (likely the outlier)
                    data[f1, pi, ki, :3] = np.nan
                    n_rejected += 1

    return data, n_rejected


# ---------------------------------------------------------------------------
# 3. Gap interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(
    data: np.ndarray,
    max_gap_frames: int = 10,
) -> tuple[np.ndarray, int]:
    """
    Fill NaN gaps of up to ``max_gap_frames`` using cubic spline
    (when ≥4 valid surrounding points) or linear interpolation.

    Returns:
        interpolated data, count of interpolated frames
    """
    from scipy.interpolate import CubicSpline

    F, P, K, _ = data.shape
    n_interpolated = 0

    for pi in range(P):
        for ki in range(K):
            for dim in range(3):  # x, y, z
                series = data[:, pi, ki, dim].copy()
                valid_mask = ~np.isnan(series)
                valid_idx = np.where(valid_mask)[0]

                if len(valid_idx) < 2:
                    continue

                # Find gap segments
                nan_mask = np.isnan(series)
                if not nan_mask.any():
                    continue

                # Identify contiguous NaN runs
                changes = np.diff(nan_mask.astype(int))
                gap_starts = np.where(changes == 1)[0] + 1
                gap_ends = np.where(changes == -1)[0] + 1

                # Handle edge cases
                if nan_mask[0]:
                    gap_starts = np.concatenate([[0], gap_starts])
                if nan_mask[-1]:
                    gap_ends = np.concatenate([gap_ends, [F]])

                for gs, ge in zip(gap_starts, gap_ends):
                    gap_len = ge - gs
                    if gap_len > max_gap_frames:
                        continue
                    # Need valid points on both sides
                    if gs == 0 or ge == F:
                        continue

                    # Gather surrounding valid points for interpolation
                    surround_idx = valid_idx[
                        (valid_idx < gs) | (valid_idx >= ge)
                    ]

                    if len(surround_idx) < 2:
                        continue

                    gap_frames = np.arange(gs, ge)

                    if len(surround_idx) >= 4:
                        # Cubic spline
                        try:
                            cs = CubicSpline(
                                surround_idx,
                                series[surround_idx],
                                extrapolate=False,
                            )
                            interp_vals = cs(gap_frames)
                            # Safety: don't extrapolate wildly
                            if not np.any(np.isnan(interp_vals)):
                                series[gap_frames] = interp_vals
                                n_interpolated += gap_len
                        except Exception:
                            # Fall back to linear
                            series[gap_frames] = np.interp(
                                gap_frames, surround_idx, series[surround_idx]
                            )
                            n_interpolated += gap_len
                    else:
                        # Linear
                        series[gap_frames] = np.interp(
                            gap_frames, surround_idx, series[surround_idx]
                        )
                        n_interpolated += gap_len

                data[:, pi, ki, dim] = series

    return data, n_interpolated


# ---------------------------------------------------------------------------
# 4. Temporal smoothing (Butterworth low-pass)
# ---------------------------------------------------------------------------

def smooth_skeleton(
    data: np.ndarray,
    fps: float = 30.0,
    cutoff_hz: float = 6.0,
    order: int = 2,
) -> tuple[np.ndarray, int]:
    """
    Apply Butterworth low-pass filter per joint per coordinate.

    Only applied to segments with ≥ ``min_segment`` consecutive valid frames.

    Returns:
        smoothed data, count of smoothed segments
    """
    from scipy.signal import butter, filtfilt

    nyq = fps / 2.0
    if cutoff_hz >= nyq:
        logger.warning(
            f"Cutoff {cutoff_hz} Hz ≥ Nyquist {nyq} Hz; clamping to {nyq * 0.9:.1f} Hz"
        )
        cutoff_hz = nyq * 0.9

    b, a = butter(order, cutoff_hz / nyq, btype="low")

    F, P, K, _ = data.shape
    min_segment = max(3 * (order + 1), 10)  # Need enough points for filtfilt
    n_smoothed = 0

    for pi in range(P):
        for ki in range(K):
            for dim in range(3):
                series = data[:, pi, ki, dim]
                valid_mask = ~np.isnan(series)

                if valid_mask.sum() < min_segment:
                    continue

                # Find contiguous valid segments
                changes = np.diff(valid_mask.astype(int))
                seg_starts = np.where(changes == 1)[0] + 1
                seg_ends = np.where(changes == -1)[0] + 1

                if valid_mask[0]:
                    seg_starts = np.concatenate([[0], seg_starts])
                if valid_mask[-1]:
                    seg_ends = np.concatenate([seg_ends, [F]])

                for ss, se in zip(seg_starts, seg_ends):
                    seg_len = se - ss
                    if seg_len >= min_segment:
                        try:
                            series[ss:se] = filtfilt(b, a, series[ss:se])
                            n_smoothed += 1
                        except Exception:
                            pass  # Skip if filtfilt fails

                data[:, pi, ki, dim] = series

    return data, n_smoothed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refine_skeleton(
    input_path: Path,
    output_path: Path | None = None,
    min_confidence: float = 0.3,
    max_reproj_px: float = 20.0,
    min_cameras: int = 2,
    max_velocity_mm: float = 300.0,
    max_gap_frames: int = 10,
    smooth_cutoff_hz: float = 6.0,
    fps: float = 30.0,
    upper_body: bool = False,
) -> Path:
    """Run the full refinement pipeline and save results."""

    logger.info(f"Loading: {input_path}")
    raw = np.load(input_path, allow_pickle=False)
    logger.info(f"  Shape: {raw.shape}")
    data = raw.copy()

    F, P, K, D = data.shape
    assert D == 7, f"Expected 7 columns, got {D}"

    total_joints = F * P * K
    initial_valid = int(np.sum(~np.isnan(data[:, :, :, IX])))
    logger.info(f"  Initial valid: {initial_valid}/{total_joints} ({initial_valid/total_joints*100:.1f}%)")

    # Stage 1: Quality gate
    logger.info("Stage 1: Quality gate")
    logger.info(f"  min_confidence={min_confidence}, max_reproj={max_reproj_px}px, min_cameras={min_cameras}")
    data, gate_stats = quality_gate(
        data,
        min_confidence=min_confidence,
        max_reproj_px=max_reproj_px,
        min_cameras=min_cameras,
        upper_body=upper_body,
    )
    post_gate = int(np.sum(~np.isnan(data[:, :, :, IX])))
    logger.info(f"  After gate: {post_gate}/{total_joints} ({post_gate/total_joints*100:.1f}%)")
    logger.info(f"  Removed: {gate_stats['removed_total']} "
                f"(conf:{gate_stats['removed_low_confidence']} "
                f"reproj:{gate_stats['removed_high_reproj']} "
                f"cams:{gate_stats['removed_few_cameras']} "
                f"body:{gate_stats['removed_lower_body']})")

    # Stage 2: Velocity outlier rejection
    logger.info("Stage 2: Velocity outlier rejection")
    logger.info(f"  max_velocity={max_velocity_mm} mm/frame")
    data, n_velocity_rejected = velocity_outlier_rejection(
        data, max_velocity_mm=max_velocity_mm, fps=fps,
    )
    post_velocity = int(np.sum(~np.isnan(data[:, :, :, IX])))
    logger.info(f"  Rejected: {n_velocity_rejected} joints")
    logger.info(f"  After velocity: {post_velocity}/{total_joints} ({post_velocity/total_joints*100:.1f}%)")

    # Stage 3: Gap interpolation
    logger.info("Stage 3: Gap interpolation")
    logger.info(f"  max_gap={max_gap_frames} frames")
    data, n_interpolated = interpolate_gaps(
        data, max_gap_frames=max_gap_frames,
    )
    post_interp = int(np.sum(~np.isnan(data[:, :, :, IX])))
    logger.info(f"  Interpolated: {n_interpolated} gap-frames (per dim)")
    logger.info(f"  After interp: {post_interp}/{total_joints} ({post_interp/total_joints*100:.1f}%)")

    # Stage 4: Temporal smoothing
    logger.info("Stage 4: Temporal smoothing (Butterworth)")
    logger.info(f"  cutoff={smooth_cutoff_hz} Hz, order=2, fps={fps}")
    data, n_smoothed = smooth_skeleton(
        data, fps=fps, cutoff_hz=smooth_cutoff_hz,
    )
    logger.info(f"  Smoothed: {n_smoothed} segments")

    final_valid = int(np.sum(~np.isnan(data[:, :, :, IX])))
    logger.info(f"RESULT: {final_valid}/{total_joints} valid ({final_valid/total_joints*100:.1f}%)")
    logger.info(f"  was {initial_valid} → gate {post_gate} → velocity {post_velocity} "
                f"→ interp {post_interp} → final {final_valid}")

    # Compute final quality stats
    valid_mask = ~np.isnan(data[:, :, :, IX])
    if valid_mask.any():
        vconf = data[:, :, :, ICONF][valid_mask]
        vrp = data[:, :, :, IREPROJ][valid_mask]
        vrp = vrp[vrp >= 0]
        vnc = data[:, :, :, INCAMS][valid_mask]
        final_stats = {
            "confidence_mean": float(np.mean(vconf)),
            "reproj_mean_px": float(np.mean(vrp)) if len(vrp) > 0 else -1,
            "n_cameras_mean": float(np.mean(vnc)),
        }
    else:
        final_stats = {}

    # Output
    if output_path is None:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_refined.npy"
    json_path = output_path.with_suffix(".json")

    np.save(output_path, data)
    logger.info(f"Saved: {output_path}")

    def _js(v):
        """Convert numpy scalars to Python natives for JSON."""
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    meta = {
        "source": str(input_path),
        "shape": [int(x) for x in data.shape],
        "parameters": {
            "min_confidence": min_confidence,
            "max_reproj_px": max_reproj_px,
            "min_cameras": min_cameras,
            "max_velocity_mm": max_velocity_mm,
            "max_gap_frames": max_gap_frames,
            "smooth_cutoff_hz": smooth_cutoff_hz,
            "fps": fps,
            "upper_body": upper_body,
        },
        "quality_gate": {k: _js(v) for k, v in gate_stats.items()},
        "velocity_rejected": _js(n_velocity_rejected),
        "interpolated_gap_frames": _js(n_interpolated),
        "smoothed_segments": _js(n_smoothed),
        "pipeline_counts": {
            "initial_valid": _js(initial_valid),
            "after_gate": _js(post_gate),
            "after_velocity": _js(post_velocity),
            "after_interpolation": _js(post_interp),
            "final_valid": _js(final_valid),
        },
        "final_quality": final_stats,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine 3D skeleton: quality filter → outlier reject → interpolate → smooth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input skeleton .npy file (F, P, K, 7)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .npy path (default: <input>_refined.npy)",
    )

    # Quality gate
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum 2D confidence (default: 0.3)")
    parser.add_argument("--max-reproj", type=float, default=20.0,
                        help="Maximum reprojection error in px (default: 20)")
    parser.add_argument("--min-cameras", type=int, default=2,
                        help="Minimum cameras required (default: 2)")

    # Velocity
    parser.add_argument("--max-velocity", type=float, default=300.0,
                        help="Max joint velocity in mm/frame (default: 300)")

    # Interpolation
    parser.add_argument("--max-gap", type=int, default=10,
                        help="Max gap length to interpolate in frames (default: 10)")

    # Smoothing
    parser.add_argument("--smooth-cutoff", type=float, default=6.0,
                        help="Butterworth cutoff frequency in Hz (default: 6.0)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frame rate of the skeleton data (default: 30)")

    # Scope
    parser.add_argument("--upper-body", action="store_true",
                        help="Keep only upper-body keypoints (head+spine+arms)")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    refine_skeleton(
        input_path=args.input,
        output_path=args.output,
        min_confidence=args.min_confidence,
        max_reproj_px=args.max_reproj,
        min_cameras=args.min_cameras,
        max_velocity_mm=args.max_velocity,
        max_gap_frames=args.max_gap,
        smooth_cutoff_hz=args.smooth_cutoff,
        fps=args.fps,
        upper_body=args.upper_body,
    )


if __name__ == "__main__":
    main()
