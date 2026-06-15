#!/usr/bin/env python3
"""Generate printable ArUco marker sheets for glasses + table setup.

Creates a single PNG image with all required markers:
- Table markers (5): large markers for table corners + centre
- Glasses markers (8): small markers for 4 glasses × 2 sides

The ``lab_dual_board`` profile generates a lab-specific setup with:
- Desk markers (6): 50mm markers on desk edges (4 corners + left/right side centre)
- Glasses markers (8): 25mm markers for 4 glasses
- Geometry export: JSON files with desk/board/camera/participant metadata

Usage:
    python tools/generate_aruco_marker_sheet.py --output markers/
    python tools/generate_aruco_marker_sheet.py --table-only --output table_markers.png
    python tools/generate_aruco_marker_sheet.py --glasses-only --output glasses_markers.png
    python tools/generate_aruco_marker_sheet.py --profile lab_dual_board --output markers/lab_dual_board

Print settings:
    - Use matte photo paper for best detection (less glare)
    - Print at 100% scale (no fit-to-page)
    - Verify printed size with ruler before attaching
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default marker configuration (matches tobii_multicam_glasses_tracker.example.yaml)
DEFAULT_ARUCO_DICT = "DICT_4X4_50"
TABLE_MARKER_IDS = [0, 1, 2, 3, 4]  # Corners + centre
GLASSES_MARKER_PAIRS = [
    ("P1", 10, 11),
    ("P2", 12, 13),
    ("P3", 14, 15),
    ("P4", 16, 17),
]

# Marker sizes in mm (for A4 printing)
TABLE_MARKER_SIZE_MM = 50
GLASSES_MARKER_SIZE_MM = 15

# Printing constants
DPI = 300
MM_PER_INCH = 25.4
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
A3_WIDTH_MM = 297
A3_HEIGHT_MM = 420

# Lab-specific preset (dual-board workflow)
LAB_PROFILE_NAME = "lab_dual_board"
LAB_TABLE_MARKER_IDS = [0, 1, 2, 3, 4, 5]  # 4 corners + left/right side centre
LAB_TABLE_MARKER_LABELS = [
    "Desk Front-Left Corner",
    "Desk Front-Right Corner",
    "Desk Back-Right Corner",
    "Desk Back-Left Corner",
    "Desk Left-Centre",
    "Desk Right-Centre",
]
LAB_GLASSES_MARKER_PAIRS = GLASSES_MARKER_PAIRS
LAB_TABLE_MARKER_SIZE_MM = 50.0
LAB_GLASSES_MARKER_SIZE_MM = 25.0

LAB_DESK_WIDTH_M = 1.80
LAB_DESK_DEPTH_M = 0.80
LAB_DESK_HEIGHT_M = 0.75
LAB_FIXED_BOARD = {
    "board_type": "7x5",
    "square_size_m": 0.069,
    "aruco_dictionary": "DICT_4X4_250",
    "origin": "board_center",
    "position_m": [0.0, 0.0, 0.0],
}


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    """Convert millimetres to pixels at given DPI."""
    return int(mm / MM_PER_INCH * dpi)


def create_aruco_dictionary(dict_name: str):
    """Create ArUco dictionary from name."""
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    dictionary_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def generate_marker(
    dictionary, marker_id: int, size_px: int, border_bits: int = 1
) -> np.ndarray:
    """Generate a single ArUco marker image.

    Returns:
        Grayscale image (size_px × size_px) with white border.
    """
    # Generate marker
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)

    # Add white border (for cutting guide and detection margin)
    border_px = size_px // 8
    bordered = np.ones((size_px + 2 * border_px, size_px + 2 * border_px), dtype=np.uint8) * 255
    bordered[border_px:border_px + size_px, border_px:border_px + size_px] = marker

    return bordered


def add_label(
    img: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.5,
) -> None:
    """Add text label to image (in-place)."""
    cv2.putText(
        img, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0), 1, cv2.LINE_AA
    )


def marker_corners_from_center(center_x_m: float, center_y_m: float, size_m: float) -> list[list[float]]:
    """Create world marker corners (TL, TR, BR, BL) for a desk-plane marker."""
    half = size_m / 2.0
    return [
        [center_x_m - half, center_y_m + half, 0.0],
        [center_x_m + half, center_y_m + half, 0.0],
        [center_x_m + half, center_y_m - half, 0.0],
        [center_x_m - half, center_y_m - half, 0.0],
    ]


def build_lab_dual_board_layout_dict() -> dict:
    """Build machine-readable lab geometry for tracker + calibration workflows."""
    half_w = LAB_DESK_WIDTH_M / 2.0
    half_d = LAB_DESK_DEPTH_M / 2.0
    marker_size_m = LAB_TABLE_MARKER_SIZE_MM / 1000.0

    # Markers are attached on desk edges; centres are on edge lines.
    table_markers = [
        {
            "id": 0,
            "label": "front_left_corner",
            "corners_m": marker_corners_from_center(-half_w, -half_d, marker_size_m),
        },
        {
            "id": 1,
            "label": "front_right_corner",
            "corners_m": marker_corners_from_center(half_w, -half_d, marker_size_m),
        },
        {
            "id": 2,
            "label": "back_right_corner",
            "corners_m": marker_corners_from_center(half_w, half_d, marker_size_m),
        },
        {
            "id": 3,
            "label": "back_left_corner",
            "corners_m": marker_corners_from_center(-half_w, half_d, marker_size_m),
        },
        {
            "id": 4,
            "label": "left_center",
            "corners_m": marker_corners_from_center(-half_w, 0.0, marker_size_m),
        },
        {
            "id": 5,
            "label": "right_center",
            "corners_m": marker_corners_from_center(half_w, 0.0, marker_size_m),
        },
    ]

    glasses = []
    for pid, left_id, right_id in LAB_GLASSES_MARKER_PAIRS:
        # Derived from user constraints:
        # - marker centre distance = 140 mm
        # - eye camera is 60 mm behind the line connecting front marker edges
        # - with 25 mm markers, centre-to-front-edge = 12.5 mm
        # -> offset magnitude from marker centre to camera along temple direction = 72.5 mm
        offset_mm = 60.0 + (LAB_GLASSES_MARKER_SIZE_MM / 2.0)
        glasses.append(
            {
                "id": f"tobii_{pid.lower()}",
                "left_marker_id": left_id,
                "right_marker_id": right_id,
                "marker_size_m": LAB_GLASSES_MARKER_SIZE_MM / 1000.0,
                "marker_center_distance_mm": 140.0,
                "camera_from_front_edge_line_mm": 60.0,
                "left_marker_offset_mm": [offset_mm, 0.0, 0.0],
                "right_marker_offset_mm": [-offset_mm, 0.0, 0.0],
                "time_offset_s": 0.0,
            }
        )

    # World frame:
    # x+: right, y+: back, z+: up, origin at fixed board centre (table centre).
    camera_height_high_m = 1.63 - LAB_DESK_HEIGHT_M
    cameras = [
        {
            "id": "cam1",
            "label": "left_front_middle",
            "position_m": [-half_w, -half_d / 2.0, camera_height_high_m],
            "focus": ["P1", "P2"],
        },
        {
            "id": "cam2",
            "label": "right_front_middle",
            "position_m": [half_w, -half_d / 2.0, camera_height_high_m],
            "focus": ["P3", "P4"],
        },
        {
            "id": "cam3",
            "label": "right_back_middle",
            "position_m": [half_w, half_d / 2.0, camera_height_high_m],
            "focus": ["P3", "P4"],
        },
        {
            "id": "cam4",
            "label": "left_back_middle",
            "position_m": [-half_w, half_d / 2.0, camera_height_high_m],
            "focus": ["P1", "P2"],
        },
        {
            "id": "cam5",
            "label": "front_center_low",
            "position_m": [0.0, -half_d, 0.0],
            "focus": ["table_overview"],
            "notes": "Cannot see front-left/front-right desk edge markers",
        },
        {
            "id": "cam6",
            "label": "back_center",
            "position_m": [0.0, half_d, camera_height_high_m],
            "focus": ["all"],
        },
        {
            "id": "cam7",
            "label": "front_center_p50",
            "position_m": [0.0, -half_d, 0.90],
            "focus": ["all"],
            "model": "panacast_50",
        },
    ]

    participants = {
        # Interpreted from user note with typo corrections:
        # P1 back-right, P2 front-right, P3 front-left, P4 back-left.
        "P1": "back_right",
        "P2": "front_right",
        "P3": "front_left",
        "P4": "back_left",
    }

    return {
        "world": {
            "coordinate_frame": {
                "origin": "fixed_7x5_board_center",
                "axes": {
                    "x": "right",
                    "y": "back",
                    "z": "up",
                },
            },
            "aruco_dictionary": DEFAULT_ARUCO_DICT,
        },
        "desk": {
            "width_m": LAB_DESK_WIDTH_M,
            "depth_m": LAB_DESK_DEPTH_M,
            "height_m": LAB_DESK_HEIGHT_M,
        },
        "fixed_board": LAB_FIXED_BOARD,
        "table_markers": table_markers,
        "glasses": glasses,
        "cameras": cameras,
        "participants": participants,
    }


def build_lab_tracker_config_dict(layout: dict) -> dict:
    """Build tracker-ready config compatible with tobii_multicam_glasses_tracker."""
    return {
        "aruco_dictionary": layout["world"]["aruco_dictionary"],
        "video_fps": 30.0,
        "min_cameras_for_triangulation": 2,
        "max_reproj_error_px": 10.0,
        "table_markers": [
            {
                "id": marker["id"],
                "comment": marker["label"],
                "corners_m": marker["corners_m"],
            }
            for marker in layout["table_markers"]
        ],
        "glasses": [
            {
                "id": glasses["id"],
                "left_marker_id": glasses["left_marker_id"],
                "right_marker_id": glasses["right_marker_id"],
                "marker_size_m": glasses["marker_size_m"],
                "left_marker_offset_mm": glasses["left_marker_offset_mm"],
                "right_marker_offset_mm": glasses["right_marker_offset_mm"],
                "gaze_ndjson": None,
                "time_offset_s": glasses["time_offset_s"],
            }
            for glasses in layout["glasses"]
        ],
        "layout_metadata": {
            "desk": layout["desk"],
            "fixed_board": layout["fixed_board"],
            "cameras": layout["cameras"],
            "participants": layout["participants"],
        },
    }


def generate_table_marker_sheet(
    dictionary,
    marker_ids: list[int],
    marker_size_mm: float,
    marker_labels: list[str] | None = None,
    cols: int = 2,
    rows: int = 3,
) -> np.ndarray:
    """Generate sheet with table markers (large format).

    Layout: 2×3 grid (5 markers + instructions)
    """
    marker_size_px = mm_to_px(marker_size_mm)
    margin_px = mm_to_px(15)

    # Calculate sheet size for configurable grid
    cell_size = marker_size_px + 2 * (marker_size_px // 8)  # marker + borders
    sheet_w = cols * cell_size + (cols + 1) * margin_px
    sheet_h = rows * cell_size + (rows + 1) * margin_px + mm_to_px(30)  # extra for title

    sheet = np.ones((sheet_h, sheet_w), dtype=np.uint8) * 255

    # Title
    cv2.putText(
        sheet, f"TABLE MARKERS ({marker_size_mm}mm)",
        (margin_px, mm_to_px(12)),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
    )

    # Place markers in grid
    y_offset = mm_to_px(25)
    for idx, marker_id in enumerate(marker_ids):
        row = idx // cols
        col = idx % cols

        x = margin_px + col * (cell_size + margin_px)
        y = y_offset + row * (cell_size + margin_px)

        marker_img = generate_marker(dictionary, marker_id, marker_size_px)
        h, w = marker_img.shape[:2]
        sheet[y:y + h, x:x + w] = marker_img

        # Label
        label_y = y + h + mm_to_px(5)
        labels = marker_labels or [
            "Corner Front-Left",
            "Corner Front-Right",
            "Corner Back-Right",
            "Corner Back-Left",
            "Table Centre",
        ]
        label_text = labels[idx] if idx < len(labels) else f"Marker {idx + 1}"
        add_label(sheet, f"ID {marker_id}: {label_text}", (x, label_y), 0.4)

    return sheet


def generate_glasses_marker_sheet(
    dictionary, marker_pairs: list[tuple[str, int, int]], marker_size_mm: float
) -> np.ndarray:
    """Generate sheet with glasses markers (small format).

    Layout: 4 rows (one per participant) × 2 columns (left + right)
    """
    marker_size_px = mm_to_px(marker_size_mm)
    margin_px = mm_to_px(10)

    # Calculate sheet size
    cell_size = marker_size_px + 2 * (marker_size_px // 8)
    label_height = mm_to_px(8)
    row_height = cell_size + label_height + margin_px

    sheet_w = 2 * cell_size + 3 * margin_px + mm_to_px(30)  # extra for row labels
    sheet_h = len(marker_pairs) * row_height + mm_to_px(30)

    sheet = np.ones((sheet_h, sheet_w), dtype=np.uint8) * 255

    # Title
    cv2.putText(
        sheet, f"GLASSES MARKERS ({marker_size_mm}mm)",
        (margin_px, mm_to_px(10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
    )

    # Column headers
    header_y = mm_to_px(20)
    x_left = mm_to_px(35)
    x_right = x_left + cell_size + margin_px
    add_label(sheet, "LEFT", (x_left + cell_size // 3, header_y), 0.5)
    add_label(sheet, "RIGHT", (x_right + cell_size // 3, header_y), 0.5)

    # Place markers
    y_offset = mm_to_px(25)
    for row_idx, (participant, left_id, right_id) in enumerate(marker_pairs):
        y = y_offset + row_idx * row_height

        # Row label
        add_label(sheet, participant, (margin_px, y + cell_size // 2), 0.6)

        # Left marker
        left_marker = generate_marker(dictionary, left_id, marker_size_px)
        h, w = left_marker.shape[:2]
        sheet[y:y + h, x_left:x_left + w] = left_marker
        add_label(sheet, f"ID {left_id}", (x_left + w // 3, y + h + mm_to_px(4)), 0.35)

        # Right marker
        right_marker = generate_marker(dictionary, right_id, marker_size_px)
        sheet[y:y + h, x_right:x_right + w] = right_marker
        add_label(sheet, f"ID {right_id}", (x_right + w // 3, y + h + mm_to_px(4)), 0.35)

    return sheet


def generate_combined_sheet(
    dictionary,
    table_ids: list[int],
    glasses_pairs: list[tuple[str, int, int]],
    table_size_mm: float,
    glasses_size_mm: float,
) -> np.ndarray:
    """Generate combined sheet with all markers on A4."""
    # Create A4 sheet
    sheet_w = mm_to_px(A4_WIDTH_MM)
    sheet_h = mm_to_px(A4_HEIGHT_MM)
    sheet = np.ones((sheet_h, sheet_w), dtype=np.uint8) * 255

    margin = mm_to_px(10)

    # Title
    cv2.putText(
        sheet, "AffectAI Tobii Glasses Tracker - Marker Set",
        (margin, mm_to_px(15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
    )

    cv2.putText(
        sheet, f"Dictionary: {DEFAULT_ARUCO_DICT}",
        (margin, mm_to_px(22)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA
    )

    # Table markers section
    y_section = mm_to_px(30)
    cv2.putText(
        sheet, f"TABLE MARKERS ({table_size_mm}mm) - Place at table corners + centre",
        (margin, y_section),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )

    table_px = mm_to_px(table_size_mm)
    y_markers = y_section + mm_to_px(5)
    spacing = table_px + mm_to_px(5)

    for i, marker_id in enumerate(table_ids):
        x = margin + i * spacing
        marker = generate_marker(dictionary, marker_id, table_px)
        h, w = marker.shape[:2]

        if x + w < sheet_w - margin:
            sheet[y_markers:y_markers + h, x:x + w] = marker
            add_label(sheet, str(marker_id), (x + w // 2 - mm_to_px(2), y_markers + h + mm_to_px(4)), 0.35)

    # Glasses markers section
    y_glasses = y_markers + table_px + mm_to_px(20)
    cv2.putText(
        sheet, f"GLASSES MARKERS ({glasses_size_mm}mm) - Attach to left & right temple",
        (margin, y_glasses),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )

    glasses_px = mm_to_px(glasses_size_mm)
    y_g_markers = y_glasses + mm_to_px(5)
    g_spacing = glasses_px + mm_to_px(8)

    for row_idx, (participant, left_id, right_id) in enumerate(glasses_pairs):
        y = y_g_markers + row_idx * (glasses_px + mm_to_px(15))
        x_label = margin
        x_left = margin + mm_to_px(15)
        x_right = x_left + g_spacing

        # Participant label
        add_label(sheet, participant, (x_label, y + glasses_px // 2), 0.4)

        # Left marker
        left_marker = generate_marker(dictionary, left_id, glasses_px)
        h, w = left_marker.shape[:2]
        sheet[y:y + h, x_left:x_left + w] = left_marker
        add_label(sheet, f"L:{left_id}", (x_left, y + h + mm_to_px(4)), 0.3)

        # Right marker
        right_marker = generate_marker(dictionary, right_id, glasses_px)
        sheet[y:y + h, x_right:x_right + w] = right_marker
        add_label(sheet, f"R:{right_id}", (x_right, y + h + mm_to_px(4)), 0.3)

    # Instructions at bottom
    y_instr = sheet_h - mm_to_px(40)
    instructions = [
        "PRINTING: Use matte paper, 100% scale, verify size with ruler",
        "TABLE: Tape markers securely at measured positions",
        "GLASSES: Attach with clear tape, avoid glare, measure offsets",
    ]
    for i, line in enumerate(instructions):
        cv2.putText(
            sheet, line,
            (margin, y_instr + i * mm_to_px(7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA
        )

    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker sheets for Tobii glasses tracking",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("markers"),
        help="Output directory or file path",
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="Generate only table markers",
    )
    parser.add_argument(
        "--glasses-only", action="store_true",
        help="Generate only glasses markers",
    )
    parser.add_argument(
        "--table-size", type=float, default=TABLE_MARKER_SIZE_MM,
        help=f"Table marker size in mm (default: {TABLE_MARKER_SIZE_MM})",
    )
    parser.add_argument(
        "--glasses-size", type=float, default=GLASSES_MARKER_SIZE_MM,
        help=f"Glasses marker size in mm (default: {GLASSES_MARKER_SIZE_MM})",
    )
    parser.add_argument(
        "--dict", default=DEFAULT_ARUCO_DICT,
        help=f"ArUco dictionary name (default: {DEFAULT_ARUCO_DICT})",
    )
    parser.add_argument(
        "--profile", default="default", choices=["default", LAB_PROFILE_NAME],
        help="Generation profile. lab_dual_board applies lab-specific marker sizes + IDs.",
    )
    parser.add_argument(
        "--export-layout", type=Path, default=None,
        help="Optional JSON path for writing machine-readable geometry metadata.",
    )
    parser.add_argument(
        "--paper-size", default="a4", choices=["a4", "a3"],
        help="Target paper size for generated sheets. A3 is recommended for large markers.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dictionary = create_aruco_dictionary(args.dict)

    table_ids = TABLE_MARKER_IDS
    glasses_pairs = GLASSES_MARKER_PAIRS
    table_size_mm = args.table_size
    glasses_size_mm = args.glasses_size
    table_labels: list[str] | None = None

    if args.profile == LAB_PROFILE_NAME:
        table_ids = LAB_TABLE_MARKER_IDS
        glasses_pairs = LAB_GLASSES_MARKER_PAIRS
        table_size_mm = LAB_TABLE_MARKER_SIZE_MM
        glasses_size_mm = LAB_GLASSES_MARKER_SIZE_MM
        table_labels = LAB_TABLE_MARKER_LABELS
        logger.info(
            "Using profile '%s': table markers=%s mm, glasses markers=%s mm",
            LAB_PROFILE_NAME,
            int(table_size_mm),
            int(glasses_size_mm),
        )

    # Determine output path(s)
    if args.output.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = None

    if args.table_only:
        logger.info(f"Generating table markers ({table_size_mm}mm)...")
        sheet = generate_table_marker_sheet(
            dictionary, table_ids, table_size_mm, table_labels
        )
        path = output_path or args.output / "table_markers.png"
        cv2.imwrite(str(path), sheet)
        logger.info(f"Saved: {path}")

    elif args.glasses_only:
        logger.info(f"Generating glasses markers ({glasses_size_mm}mm)...")
        sheet = generate_glasses_marker_sheet(
            dictionary, glasses_pairs, glasses_size_mm
        )
        path = output_path or args.output / "glasses_markers.png"
        cv2.imwrite(str(path), sheet)
        logger.info(f"Saved: {path}")

    elif args.profile == LAB_PROFILE_NAME:
        logger.info(
            "Skipping combined sheet for profile '%s' (profile emits dedicated sheets).",
            LAB_PROFILE_NAME,
        )
        output_dir = args.output if output_path is None else output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.paper_size == "a3":
            # Keep profile output as two A3 desk pages (3 markers/page) to match
            # existing print/cut workflow while preserving true marker size.
            page1_ids = table_ids[:3]
            page1_labels = table_labels[:3] if table_labels else None
            page1 = generate_table_marker_sheet(
                dictionary,
                page1_ids,
                table_size_mm,
                page1_labels,
                cols=2,
                rows=2,
            )
            page1_path = output_dir / "table_markers_a3_page1.png"
            cv2.imwrite(str(page1_path), page1)
            logger.info(f"Saved: {page1_path}")

            page2_ids = table_ids[3:]
            page2_labels = table_labels[3:] if table_labels else None
            page2 = generate_table_marker_sheet(
                dictionary,
                page2_ids,
                table_size_mm,
                page2_labels,
                cols=2,
                rows=2,
            )
            page2_path = output_dir / "table_markers_a3_page2.png"
            cv2.imwrite(str(page2_path), page2)
            logger.info(f"Saved: {page2_path}")
        else:
            table_sheet = generate_table_marker_sheet(
                dictionary, table_ids, table_size_mm, table_labels
            )
            table_path = output_dir / "table_markers.png"
            cv2.imwrite(str(table_path), table_sheet)
            logger.info(f"Saved: {table_path}")

        glasses_sheet = generate_glasses_marker_sheet(
            dictionary, glasses_pairs, glasses_size_mm
        )
        glasses_suffix = "_a3" if args.paper_size == "a3" else ""
        glasses_path = output_dir / f"glasses_markers{glasses_suffix}.png"
        cv2.imwrite(str(glasses_path), glasses_sheet)
        logger.info(f"Saved: {glasses_path}")

    else:
        # Generate combined A4 sheet
        logger.info("Generating combined marker sheet (A4)...")
        sheet = generate_combined_sheet(
            dictionary,
            table_ids,
            glasses_pairs,
            table_size_mm,
            glasses_size_mm,
        )
        path = output_path or args.output / "aruco_marker_sheet.png"
        cv2.imwrite(str(path), sheet)
        logger.info(f"Saved: {path}")

        # Also generate individual sheets
        table_sheet = generate_table_marker_sheet(
            dictionary, table_ids, table_size_mm, table_labels
        )
        table_path = args.output / "table_markers.png"
        cv2.imwrite(str(table_path), table_sheet)
        logger.info(f"Saved: {table_path}")

        glasses_sheet = generate_glasses_marker_sheet(
            dictionary, glasses_pairs, glasses_size_mm
        )
        glasses_path = args.output / "glasses_markers.png"
        cv2.imwrite(str(glasses_path), glasses_sheet)
        logger.info(f"Saved: {glasses_path}")

    if args.profile == LAB_PROFILE_NAME:
        layout_path = args.export_layout or (
            (args.output if args.output.is_dir() else args.output.parent)
            / "lab_dual_board_layout.json"
        )
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout = build_lab_dual_board_layout_dict()
        layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
        logger.info(f"Saved: {layout_path}")

        tracker_cfg_path = layout_path.parent / "tobii_multicam_glasses_tracker_lab.json"
        tracker_cfg = build_lab_tracker_config_dict(layout)
        tracker_cfg_path.write_text(json.dumps(tracker_cfg, indent=2), encoding="utf-8")
        logger.info(f"Saved: {tracker_cfg_path}")

    logger.info("\nDone! Print at 100% scale on matte paper.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
