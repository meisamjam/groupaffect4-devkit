#!/usr/bin/env python3
"""
Re-centre a multi-camera calibration TOML on a chosen reference camera.

The anipose/FreeMoCap calibration pipeline picks whichever camera was
``cam_0`` as the world-frame origin.  In our lab the four Jabra PanaCast 20
cameras are mounted **upside-down**, so having a P20 as the origin produces
a world frame that is rotated 180° around the optical axis.

This tool:

1.  Reads the calibration TOML (anipose format).
2.  Re-expresses every camera's extrinsics so that the chosen reference
    camera becomes the new origin (identity rotation, zero translation).
3.  Optionally applies a 180° image-flip correction to specified cameras
    (``--flip-cameras``).  When enabled the script:
        a. rotates each flagged camera's extrinsic R by diag(-1,-1,1)
           (negates camera-X and camera-Y, equivalent to a 180° rotation
           of the image around the principal point), and
        b. similarly transforms its translation vector.
    The **keypoint coordinates** produced by the detector must be flipped
    the same way (``(x, y) → (W-1-x, H-1-y)``) — this is handled by
    ``multicam_pose3d.py --flip-cameras``.
4.  Writes a new TOML file.

Math
----
Current extrinsics:      ``X_cam_i = R_i · X_world + t_i``
Reference cam (cam_ref): ``X_cam_ref = R_ref · X_world + t_ref``

We want the new world frame to equal the camera frame of cam_ref:
    X_world_new = R_ref · X_world_old + t_ref
    ⇒  X_world_old = R_ref^T · (X_world_new − t_ref)

Substituting into the i-th camera's equation:
    X_cam_i = R_i · R_ref^T · X_world_new  +  (t_i − R_i · R_ref^T · t_ref)

So:
    R_i_new = R_i · R_ref^T
    t_i_new = t_i − R_i · R_ref^T · t_ref

For cam_ref: R_new = I, t_new = 0.  ✓

Usage
-----
    # Make P50 (cam_4) the origin
    python tools/recenter_calibration.py \\
        --input  new_data/ses-20260202_test/video/video_camera_calibration.toml \\
        --output new_data/ses-20260202_test/video/video_camera_calibration_p50.toml \\
        --reference cam_4

    # Also flip P20 cameras (cam_0 .. cam_3)
    python tools/recenter_calibration.py \\
        --input  new_data/ses-20260202_test/video/video_camera_calibration.toml \\
        --output new_data/ses-20260202_test/video/video_camera_calibration_p50.toml \\
        --reference cam_4 \\
        --flip-cameras cam_0 cam_1 cam_2 cam_3
"""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        import toml  # type: ignore[import-untyped]
        with open(path) as f:
            return toml.load(f)


def _dump_toml(data: dict, path: Path) -> None:
    """Write calibration dict to TOML (toml or tomli_w)."""
    try:
        import tomli_w  # type: ignore[import-untyped]
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
        return
    except ImportError:
        pass
    try:
        import toml  # type: ignore[import-untyped]
        with open(path, "w") as f:
            toml.dump(data, f)
        return
    except ImportError:
        pass
    # Fallback: manual TOML writer for the simple flat structure
    _write_toml_manual(data, path)


def _fmt(v: float) -> str:
    """Format a float: strip trailing zeros but keep at least one decimal."""
    s = f"{v:.15g}"
    return s


def _write_toml_manual(data: dict, path: Path) -> None:
    """Manual TOML writer for anipose-style calibration files."""
    lines: list[str] = []
    for section in sorted(data.keys()):
        sd = data[section]
        if not isinstance(sd, dict):
            continue
        lines.append(f"[{section}]")
        for key, val in sd.items():
            if isinstance(val, str):
                lines.append(f'{key} = "{val}"')
            elif isinstance(val, bool):
                lines.append(f"{key} = {'true' if val else 'false'}")
            elif isinstance(val, (int, float)):
                lines.append(f"{key} = {_fmt(val)}")
            elif isinstance(val, list):
                lines.append(f"{key} = {_list_to_toml(val)}")
            else:
                lines.append(f"{key} = {val!r}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _list_to_toml(lst: list) -> str:
    """Recursively format a list for TOML."""
    if not lst:
        return "[]"
    if isinstance(lst[0], list):
        inner = ", ".join(_list_to_toml(sub) for sub in lst)
        return f"[ {inner},]"
    # Flat list of numbers
    elems = ", ".join(_fmt(v) if isinstance(v, float) else str(v) for v in lst)
    return f"[ {elems},]"


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def _rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues vector → 3×3 rotation matrix."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rvec).as_matrix()


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → Rodrigues vector."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


# 180° flip matrix: negate X and Y in camera frame
FLIP_180 = np.diag([-1.0, -1.0, 1.0])


def recenter_calibration(
    calib: dict,
    reference_key: str,
    flip_cameras: list[str] | None = None,
) -> dict:
    """Re-centre calibration on *reference_key* camera.

    Parameters
    ----------
    calib : dict
        Parsed calibration TOML.
    reference_key : str
        Camera key to become the new origin (e.g. ``cam_4``).
    flip_cameras : list[str] | None
        Camera keys whose extrinsics should be rotated 180° around the
        optical axis (to correct for upside-down mounting).  Applied
        **before** the re-centering transform.

    Returns
    -------
    dict
        New calibration dict with transformed extrinsics.
    """
    out = copy.deepcopy(calib)
    flip_cameras = flip_cameras or []

    # ---- Step 0: Apply 180° flip to flagged cameras -----------------------
    for ck in flip_cameras:
        if ck not in out:
            logger.warning(f"Flip camera {ck!r} not found in calibration — skipping")
            continue
        cd = out[ck]
        R = _rotvec_to_matrix(np.array(cd["rotation"]))
        t = np.array(cd["translation"])
        R_new = FLIP_180 @ R
        t_new = FLIP_180 @ t
        cd["rotation"] = R_new  # store as matrix temporarily
        cd["translation"] = t_new
        cd["_flipped"] = True
        logger.info(f"  Applied 180° flip to {ck}")

    # ---- Step 1: Get reference camera's R and t ---------------------------
    ref = out[reference_key]
    if isinstance(ref["rotation"], np.ndarray) and ref["rotation"].ndim == 2:
        R_ref = ref["rotation"]
    else:
        R_ref = _rotvec_to_matrix(np.array(ref["rotation"]))
    t_ref = np.array(ref["translation"]).reshape(3)

    R_ref_T = R_ref.T  # R_ref^T

    # ---- Step 2: Transform every camera -----------------------------------
    cam_keys = [k for k in out if k.startswith("cam_")]
    for ck in cam_keys:
        cd = out[ck]
        if isinstance(cd["rotation"], np.ndarray) and cd["rotation"].ndim == 2:
            R_i = cd["rotation"]
        else:
            R_i = _rotvec_to_matrix(np.array(cd["rotation"]))
        t_i = np.array(cd["translation"]).reshape(3)

        # New extrinsics
        R_new = R_i @ R_ref_T
        t_new = t_i - R_i @ R_ref_T @ t_ref

        # Rodrigues vector
        rvec_new = _matrix_to_rotvec(R_new)

        # World orientation & position
        world_R = R_new.T
        world_pos = -world_R @ t_new

        # Store
        cd["rotation"] = rvec_new.tolist()
        cd["translation"] = t_new.tolist()
        cd["world_orientation"] = world_R.tolist()
        cd["world_position"] = world_pos.tolist()

    # ---- Step 3: Clean up temporary flags ---------------------------------
    for ck in cam_keys:
        out[ck].pop("_flipped", None)

    # ---- Step 4: Update metadata ------------------------------------------
    if "metadata" not in out:
        out["metadata"] = {}
    out["metadata"]["reference_camera"] = reference_key
    out["metadata"]["recentered_at"] = datetime.datetime.now().isoformat(
        timespec="seconds"
    )
    if flip_cameras:
        out["metadata"]["flipped_cameras"] = flip_cameras

    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_recenter(original: dict, recentered: dict, ref_key: str) -> None:
    """Print before/after comparison for verification."""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    # Reference camera should have identity R and zero t
    ref = recentered[ref_key]
    rvec = np.array(ref["rotation"])
    tvec = np.array(ref["translation"])
    logger.info(f"\n  Reference ({ref_key}):")
    logger.info(f"    rotation    = {rvec}  (should be ~[0,0,0])")
    logger.info(f"    translation = {tvec}  (should be ~[0,0,0])")
    logger.info(f"    |rvec|={np.linalg.norm(rvec):.6f}  |tvec|={np.linalg.norm(tvec):.6f}")

    # For each camera, show world position before and after
    cam_keys = sorted(k for k in original if k.startswith("cam_"))
    logger.info(f"\n  {'Camera':<8} {'Original world_pos':>40}  {'New world_pos':>40}")
    logger.info(f"  {'------':<8} {'------------------':>40}  {'-----------':>40}")
    for ck in cam_keys:
        old_wp = np.array(original[ck]["world_position"])
        new_wp = np.array(recentered[ck]["world_position"])
        logger.info(
            f"  {ck:<8} [{old_wp[0]:8.1f}, {old_wp[1]:8.1f}, {old_wp[2]:8.1f}]"
            f"              [{new_wp[0]:8.1f}, {new_wp[1]:8.1f}, {new_wp[2]:8.1f}]"
        )

    # Verify: pair-wise distances should be preserved
    logger.info(f"\n  Pair-wise distance check (should be unchanged):")
    for i, ck_a in enumerate(cam_keys):
        for ck_b in cam_keys[i + 1 :]:
            old_a = np.array(original[ck_a]["world_position"])
            old_b = np.array(original[ck_b]["world_position"])
            new_a = np.array(recentered[ck_a]["world_position"])
            new_b = np.array(recentered[ck_b]["world_position"])
            d_old = np.linalg.norm(old_a - old_b)
            d_new = np.linalg.norm(new_a - new_b)
            delta = abs(d_old - d_new)
            ok = "✓" if delta < 0.1 else "✗"
            logger.info(
                f"    {ck_a}-{ck_b}: {d_old:.1f} → {d_new:.1f}  Δ={delta:.4f} {ok}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Re-centre multi-camera calibration on a reference camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Input calibration TOML",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output (re-centred) calibration TOML",
    )
    parser.add_argument(
        "--reference", type=str, default="cam_4",
        help="Camera key to use as new world origin (default: cam_4 = P50)",
    )
    parser.add_argument(
        "--flip-cameras", nargs="*", metavar="CAM",
        help="Camera keys to apply 180° image-flip correction "
             "(e.g. cam_0 cam_1 cam_2 cam_3 for upside-down P20s)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    logger.info(f"Input:     {args.input}")
    logger.info(f"Reference: {args.reference}")
    if args.flip_cameras:
        logger.info(f"Flip:      {args.flip_cameras}")

    original = _load_toml(args.input)

    if args.reference not in original:
        logger.error(
            f"Reference camera {args.reference!r} not found.  "
            f"Available: {[k for k in original if k.startswith('cam_')]}"
        )
        sys.exit(1)

    recentered = recenter_calibration(
        original,
        reference_key=args.reference,
        flip_cameras=args.flip_cameras,
    )

    verify_recenter(original, recentered, args.reference)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _dump_toml(recentered, args.output)
    logger.info(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
