#!/usr/bin/env python3
"""Merge two camera calibration TOML files.

Takes camera entries from --new-toml (higher-quality calibration) and
transplants any cameras missing from it (by name match) from --old-toml.
Useful when a new calibration session only captured a subset of cameras.

Usage
-----
    python tools/merge_calibration_tomls.py \\
        --new-toml calibration_charuco_20260311_p20only.toml \\
        --old-toml calibration_charuco.toml \\
        --output   calibration_charuco_merged.toml

The substituted cameras from --old-toml are flagged with
``source = "old_toml"`` in the merged output metadata section.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _load_toml(path: Path) -> dict:
    """Load a TOML file using either tomllib (3.11+) or the toml package."""
    try:
        import tomllib  # type: ignore  # Python 3.11+

        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import toml  # type: ignore

        return toml.load(path)
    except ImportError:
        pass
    # Manual minimal TOML parser for key=value and sections
    return _parse_toml_minimal(path)


def _parse_toml_minimal(path: Path) -> dict:
    """Very minimal TOML parser sufficient for anipose calibration files."""
    import ast

    result: dict = {}
    current_section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and not line.startswith("[["):
            current_section = line.strip("[]")
            result.setdefault(current_section, {})
        elif "=" in line and current_section is not None:
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            try:
                result[current_section][k] = ast.literal_eval(v)
            except Exception:
                result[current_section][k] = v
        elif "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            try:
                result[k] = ast.literal_eval(v)
            except Exception:
                result[k] = v

    return result


def _dump_toml(data: dict, path: Path) -> None:
    """Write a TOML file. Tries toml package first, then manual serialization."""
    try:
        import toml  # type: ignore

        path.write_text(toml.dumps(data), encoding="utf-8")
        return
    except ImportError:
        pass
    # Manual serialization for anipose calibration structure
    lines: list[str] = []
    for key, val in data.items():
        if isinstance(val, dict):
            lines.append(f"[{key}]")
            for k, v in val.items():
                lines.append(f"{k} = {_toml_value(v)}")
            lines.append("")
        else:
            lines.append(f"{key} = {_toml_value(val)}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        inner = ", ".join(_toml_value(i) for i in v)
        return f"[ {inner},]"
    return repr(v)


def _cam_keys(data: dict) -> list[str]:
    return sorted(k for k in data if re.match(r"cam_\d+$", k))


def _normalise_cam_name(name: str) -> str:
    """Strip trailing _video suffix for comparison."""
    return name.removesuffix("_video")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-toml", required=True, type=Path, help="Higher-quality (partial) calibration TOML")
    parser.add_argument("--old-toml", required=True, type=Path, help="Fallback calibration TOML for missing cameras")
    parser.add_argument("--output", required=True, type=Path, help="Output merged TOML path")
    args = parser.parse_args()

    new_data = _load_toml(args.new_toml)
    old_data = _load_toml(args.old_toml)

    new_cams = _cam_keys(new_data)
    old_cams = _cam_keys(old_data)

    # Build normalised name → section key maps
    new_by_name = {_normalise_cam_name(new_data[k]["name"]): k for k in new_cams}
    old_by_name = {_normalise_cam_name(old_data[k]["name"]): k for k in old_cams}

    print("=== NEW TOML cameras (keeping as-is) ===")
    for name, key in sorted(new_by_name.items()):
        cam = new_data[key]
        fx = cam["matrix"][0][0]
        k1 = cam["distortions"][0]
        print(f"  {name:<40s}  fx={fx:.1f}  k1={k1:+.4f}")

    print()
    print("=== OLD TOML cameras checked for transplantation ===")
    transplanted: list[str] = []
    for norm_name, old_key in sorted(old_by_name.items()):
        if norm_name in new_by_name:
            print(f"  {norm_name:<40s}  → already in new TOML, skipping")
        else:
            print(f"  {norm_name:<40s}  → TRANSPLANTING from old TOML")
            transplanted.append(norm_name)

    # Build merged dict: new cam_0..cam_N, then append transplanted cams
    merged: dict = {}

    # Copy new cameras (renumbered 0-based)
    for idx, key in enumerate(new_cams):
        merged[f"cam_{idx}"] = dict(new_data[key])

    next_idx = len(new_cams)
    for norm_name in sorted(transplanted):
        old_key = old_by_name[norm_name]
        entry = dict(old_data[old_key])
        # Normalise name to match calibration video stem naming convention
        # (keep original name from old TOML — auto-mapper handles both styles)
        merged[f"cam_{next_idx}"] = entry
        next_idx += 1

    # Metadata: merge from new, note transplanted cameras
    meta = dict(new_data.get("metadata", {}))
    meta["transplanted_cameras"] = transplanted
    meta["new_toml_source"] = str(args.new_toml)
    meta["old_toml_source"] = str(args.old_toml)
    merged["metadata"] = meta

    _dump_toml(merged, args.output)
    print()
    print(f"Written {args.output}")
    print(f"  Total cameras : {next_idx}")
    print(f"  From new TOML : {len(new_cams)} cameras (cam_0 to cam_{len(new_cams)-1})")
    print(f"  Transplanted  : {len(transplanted)} cameras from old TOML")
    if transplanted:
        print(f"  NOTE: Transplanted cameras have approximate extrinsics from the old")
        print(f"        calibration. Re-run calibration with board in their FoV to improve.")


if __name__ == "__main__":
    main()
