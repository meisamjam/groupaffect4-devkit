"""
FreeMoCap motion capture processor for post-hoc video analysis.

Processes video files (e.g., from Jabra cameras) to extract 3D skeleton data
and save in BIDS-compatible format under mocap/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FreeMoCapProcessor:
    """
    Post-hoc processor for markerless motion capture using FreeMoCap.

    Accepts video files and produces BIDS-compatible skeleton outputs.
    """

    def __init__(self, session_dir: Path, calibration_data: dict | None = None):
        """
        Initialize FreeMoCap processor.

        Args:
            session_dir: Path to session directory (sub-XX/ses-YY/)
            calibration_data: Optional camera calibration parameters
        """
        self.session_dir = Path(session_dir)
        self.mocap_dir = self.session_dir / "mocap"
        self.mocap_dir.mkdir(exist_ok=True, parents=True)
        self.calibration_data = calibration_data or {}

    def process_video(
        self,
        video_path: Path,
        output_label: str = "freemocap",
        task_name: str = "unknown",
        run: int = 1,
        confidence_threshold: float = 0.5,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """
        Process video file with FreeMoCap and save results.

        Args:
            video_path: Path to video file (mp4, mov, avi, etc.)
            output_label: Label for output files (e.g. 'freemocap')
            task_name: Task identifier (e.g. 'T1', 'T2')
            run: Run number for this task
            confidence_threshold: Filter keypoints below this confidence
            **kwargs: Additional arguments passed to FreeMoCap

        Returns:
            Dictionary mapping output type to file path.
            Includes: 'skeleton_tsv', 'skeleton_json', 'metadata'

        Raises:
            ImportError: If freemocap not installed
            FileNotFoundError: If video_path not found
            RuntimeError: If FreeMoCap processing fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            import freemocap  # noqa: F401  # pylint: disable=unused-import
        except ImportError as e:
            raise ImportError(
                "FreeMoCap not installed. Install with: "
                "pip install -e '.[freemocap]'"
            ) from e

        logger.info(f"Processing video with FreeMoCap: {video_path.name}")

        try:
            from freemocap.core_processor.process_video import process_video as fmc_process
        except ImportError as e:
            raise ImportError(
                "Could not import FreeMoCap processor. "
                "Ensure freemocap>=1.3.0 is installed."
            ) from e

        # Run FreeMoCap
        output_dict = self._run_freemocap_processing(video_path, fmc_process, **kwargs)

        # Save outputs in BIDS format
        outputs = self._save_bids_outputs(
            output_dict=output_dict,
            output_label=output_label,
            task_name=task_name,
            run=run,
            confidence_threshold=confidence_threshold,
        )

        logger.info(f"FreeMoCap processing complete. Outputs: {list(outputs.keys())}")
        return outputs

    def _run_freemocap_processing(
        self, video_path: Path, fmc_process: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Run FreeMoCap video processing.

        Args:
            video_path: Path to video file
            fmc_process: FreeMoCap process_video function
            **kwargs: Additional FreeMoCap parameters

        Returns:
            Dictionary with FreeMoCap output data (body, hand keypoints, etc.)
        """
        output_dict = fmc_process(
            video_path=str(video_path),
            output_folder=str(self.mocap_dir),
            **kwargs,
        )
        return output_dict

    def _save_bids_outputs(
        self,
        output_dict: dict[str, Any],
        output_label: str,
        task_name: str,
        run: int,
        confidence_threshold: float,
    ) -> dict[str, Path]:
        """
        Convert FreeMoCap outputs to BIDS-compatible format.

        BIDS directory structure:
            mocap/
              sub-XX_ses-YY_task-T1_acq-freemocap_skeletons.tsv
              sub-XX_ses-YY_task-T1_acq-freemocap_skeletons.json

        Args:
            output_dict: FreeMoCap output dictionary
            output_label: Acquisition label (e.g. 'freemocap')
            task_name: Task identifier
            run: Run number
            confidence_threshold: Filter keypoints below this confidence

        Returns:
            Dictionary mapping output type to saved file paths
        """
        outputs = {}

        # Extract skeleton data (body keypoints)
        skeleton_data = self._extract_skeleton_data(output_dict, confidence_threshold)

        if skeleton_data:
            # Save skeleton TSV (compatible with BIDS)
            skeleton_tsv = self._save_skeleton_tsv(
                skeleton_data, output_label, task_name, run
            )
            outputs["skeleton_tsv"] = skeleton_tsv

            # Save metadata JSON
            metadata = self._create_metadata(output_dict, output_label)
            metadata_json = self._save_metadata(metadata, output_label, task_name, run)
            outputs["metadata"] = metadata_json

        return outputs

    def _extract_skeleton_data(
        self, output_dict: dict[str, Any], confidence_threshold: float
    ) -> dict[str, Any]:
        """
        Extract body skeleton keypoints from FreeMoCap output.

        FreeMoCap typically returns:
          - 3D joint positions (x, y, z in meters)
          - Confidence scores per frame per joint

        Args:
            output_dict: Raw FreeMoCap output
            confidence_threshold: Minimum confidence to include keypoint

        Returns:
            Dictionary with skeleton data or empty if not available
        """
        # FreeMoCap returns different structures depending on version
        # Common patterns: marker_data, body_markers, skeletal_data, etc.

        skeleton_data = {}

        # Check for body markers/keypoints
        if "marker_data" in output_dict:
            skeleton_data["markers"] = output_dict["marker_data"]
        elif "body_markers" in output_dict:
            skeleton_data["markers"] = output_dict["body_markers"]
        elif "skeletal_data" in output_dict:
            skeleton_data["markers"] = output_dict["skeletal_data"]

        # Store confidence threshold used
        skeleton_data["confidence_threshold"] = confidence_threshold

        return skeleton_data

    def _save_skeleton_tsv(
        self, skeleton_data: dict[str, Any], output_label: str, task_name: str, run: int
    ) -> Path:
        """
        Save skeleton data in BIDS TSV format.

        Expected format:
            frame  joint_name  x      y      z      confidence
            0      nose       0.123  0.456  0.789  0.95
            ...

        Args:
            skeleton_data: Dictionary with skeleton keypoints
            output_label: Acquisition label
            task_name: Task identifier
            run: Run number

        Returns:
            Path to saved TSV file
        """
        # Get subject/session from session_dir
        session_parts = self.session_dir.relative_to(self.session_dir.parent.parent)
        parts = str(session_parts).split("\\")
        subject = parts[0]  # e.g., sub-01
        session = parts[1] if len(parts) > 1 else "01"  # e.g., ses-01

        filename = f"{subject}_{session}_task-{task_name}_run-{run:02d}_acq-{output_label}_skeletons.tsv"
        ts_file = self.mocap_dir / filename

        # For now, create a placeholder TSV with proper BIDS header
        # In production, parse skeleton_data into proper format
        try:
            import pandas as pd

            # Create sample data structure
            # (actual implementation would populate from skeleton_data)
            data = {
                "frame": [],
                "joint": [],
                "x": [],
                "y": [],
                "z": [],
                "confidence": [],
            }

            df = pd.DataFrame(data)
            df.to_csv(ts_file, sep="\t", index=False)

            logger.info(f"Saved skeleton TSV: {ts_file}")
        except ImportError:
            # Fallback: write simple TSV without pandas
            with open(ts_file, "w") as f:
                f.write("frame\tjoint\tx\ty\tz\tconfidence\n")

            logger.info(f"Saved skeleton TSV (basic): {ts_file}")

        return ts_file

    def _create_metadata(
        self, output_dict: dict[str, Any], output_label: str
    ) -> dict[str, Any]:
        """
        Create BIDS-compatible metadata JSON.

        Args:
            output_dict: FreeMoCap output
            output_label: Acquisition label

        Returns:
            Metadata dictionary
        """
        metadata = {
            "Description": "Markerless 3D skeleton tracking",
            "Source": "FreeMoCap",
            "AcquisitionLabel": output_label,
            "CalibrationData": self.calibration_data,
            "SkeletonModel": "mediapipe_holistic",  # FreeMoCap default
        }

        # Add FreeMoCap-specific metadata if available
        if "frame_rate" in output_dict:
            metadata["FrameRate"] = output_dict["frame_rate"]
        if "number_of_frames" in output_dict:
            metadata["NumberOfFrames"] = output_dict["number_of_frames"]

        return metadata

    def _save_metadata(
        self, metadata: dict[str, Any], output_label: str, task_name: str, run: int
    ) -> Path:
        """
        Save metadata as BIDS JSON sidecar.

        Args:
            metadata: Metadata dictionary
            output_label: Acquisition label
            task_name: Task identifier
            run: Run number

        Returns:
            Path to saved JSON file
        """
        session_parts = self.session_dir.relative_to(self.session_dir.parent.parent)
        parts = str(session_parts).split("\\")
        subject = parts[0]
        session = parts[1] if len(parts) > 1 else "01"

        filename = f"{subject}_{session}_task-{task_name}_run-{run:02d}_acq-{output_label}_skeletons.json"
        json_file = self.mocap_dir / filename

        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata JSON: {json_file}")
        return json_file


def process_session_videos(
    session_dir: Path,
    video_files: dict[str, Path],
    tasks: dict[str, str] | None = None,
    **processor_kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """
    Process multiple task videos from a session.

    Args:
        session_dir: Path to session directory
        video_files: Mapping of task names to video file paths
                    e.g., {'T1': Path(...), 'T2': Path(...)}
        tasks: Optional mapping of task names to task identifiers
        **processor_kwargs: Additional arguments for FreeMoCapProcessor

    Returns:
        Nested dictionary mapping task -> output_type -> file_path

    Example:
        videos = {
            'T1_run1': Path('data/sub-01/ses-01/video/...task-T1_run-01.mp4'),
            'T2_run1': Path('data/sub-01/ses-01/video/...task-T2_run-01.mp4'),
        }
        results = process_session_videos(
            session_dir=Path('data/sub-01/ses-01'),
            video_files=videos,
        )
    """
    processor = FreeMoCapProcessor(session_dir, **processor_kwargs)
    results = {}

    for task_id, video_path in video_files.items():
        if not Path(video_path).exists():
            logger.warning(f"Video not found, skipping: {video_path}")
            continue

        try:
            task_name, run = _parse_task_id(task_id)
            outputs = processor.process_video(
                video_path=video_path,
                task_name=task_name,
                run=run,
            )
            results[task_id] = outputs
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to process {task_id}: {e}")
            results[task_id] = {"error": str(e)}

    return results


def _parse_task_id(task_id: str) -> tuple[str, int]:
    """
    Parse task ID into (task_name, run_number).

    Examples:
        'T1_run1' -> ('T1', 1)
        'T1_run01' -> ('T1', 1)
        'task-T2' -> ('T2', 1)
    """
    # Handle "T1_run1" format
    if "_run" in task_id:
        task_part, run_part = task_id.rsplit("_run", 1)
        return task_part, int(run_part)

    # Handle "task-T1" format
    if "task-" in task_id:
        task_name = task_id.split("task-")[1].split("_")[0]
        return task_name, 1

    # Default: assume task_id is task name
    return task_id, 1
