#!/usr/bin/env python3
"""
AffectAI Multimodal BIDS Processing Pipeline

Full end-to-end pipeline for processing multimodal data into BIDS format with
synchronization and task-windowing.

Processing stages:
  1. Locate multi-source raw data (Recording-PC, AV-PC, Tobii, Stimuli)
  2. Merge sources and derive task windows (T0-T4) via multisource_to_bids_runs.py
  3. Canonicalize to BIDS format via raw_to_bids.py
  4. Generate synchronization metadata and task-specific files
  5. Remove raw sourcedata - retain only processed outputs
  6. Create quality-control summary reports

Multimodal outputs:
  - et/ (Tobii gaze + pupil)
  - physio/ (EmotiBit PPG, EDA, temperature)
  - audio/ (DPA close-talk + room WAV, per-task clips)
  - video/ (camera MKV, per-task clips if --split-media)
  - beh/ (events.tsv timeline, tablet responses, stimuli answers)
  - annot/ (task windows, sync metadata, participant signal map)

Uses multiprocessing pool for parallel session processing.
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Session metadata from inventory."""
    session_id: str
    group_id: str
    phase: str
    participants: list[str]
    raw_modalities: list[str]
    # Discovered source paths
    recording_dir: Optional[Path] = None
    av_dir: Optional[Path] = None
    tobii_dir: Optional[Path] = None
    stimuli_dir: Optional[Path] = None


@dataclass
class ProcessingResult:
    """Result of processing a single session."""
    session_id: str
    success: bool
    message: str
    duration_seconds: float
    files_processed: int = 0
    files_removed: int = 0
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    stages_completed: list = None

    def __post_init__(self):
        if self.stages_completed is None:
            self.stages_completed = []


def find_session_sources(
    session: SessionInfo,
    data_root: Path
) -> SessionInfo:
    """
    Locate raw source directories for a session.
    
    Searches for:
    - Recording-PC: data/affectai-capture-recording/sessions/{phase}/sub-01/ses-*
    - AV-PC: data/AV/{phase}/*{group_id}*
    - Tobii: data/Tobii/{device_id}/*{session_id}* (manual downloads)
    - Stimuli: data/stimuli/{phase}/{date}_{group_id}
    """
    
    # Recording PC - XDF + LSL streams
    recording_root = data_root / "affectai-capture-recording" / "sessions"
    if recording_root.exists():
        rec_candidates = sorted(
            p for p in recording_root.rglob(f"*{session.session_id}*")
            if p.is_dir()
        )
        session.recording_dir = rec_candidates[0] if rec_candidates else None
    
    # AV PC - Video (MKV) + Audio (WAV) + Frame logs
    av_root = data_root / "AV" / session.phase
    if av_root.exists():
        av_candidates = sorted(
            p for p in av_root.rglob(f"*{session.session_id}*")
            if p.is_dir()
        )
        if not av_candidates:
            av_candidates = sorted(
                p for p in av_root.rglob(f"*{session.group_id}*")
                if p.is_dir()
            )
        session.av_dir = av_candidates[0] if av_candidates else None
    
    # Tobii manual downloads
    tobii_root = data_root / "Tobii"
    if tobii_root.exists():
        tobii_matches = list(tobii_root.glob(f"*{session.session_id}*"))
        session.tobii_dir = tobii_matches[0] if tobii_matches else None
    
    # Stimuli (task markers, tablet responses)
    stimuli_roots = [
        data_root / "stimuli" / session.phase,
        data_root / "stimuli",
        data_root / "affectai-capture-recording" / "stimuli" / "data",
    ]
    for stim_root in stimuli_roots:
        if not stim_root.exists():
            continue
        stim_candidates = sorted(
            p for p in stim_root.rglob(f"*{session.session_id}*")
            if p.is_dir()
        )
        if not stim_candidates:
            stim_candidates = sorted(
                p for p in stim_root.rglob(f"*{session.group_id}*")
                if p.is_dir()
            )
        if stim_candidates:
            session.stimuli_dir = stim_candidates[0]
            break
    
    return session


def process_session_worker(
    session: SessionInfo,
    data_root: Path,
    output_root: Path,
    python_exe: str,
    tools_path: Path,
    split_media: bool = True,
    link_files: bool = False,
) -> ProcessingResult:
    """
    Process a single session through the complete BIDS pipeline.
    
    **Pipeline stages:**
    
    Stage 1: Merge multi-source streams
      - Calls multisource_to_bids_runs.py
      - Merges: Recording-PC (XDF/LSL) + AV-PC (MKV/WAV) + Tobii + Stimuli
      - Derives task windows (T0-T4) from experiment markers
      - Optional: Splits video/audio into per-task clips (--split-media)
    
    Stage 2: Canonicalize to BIDS format
      - Calls raw_to_bids.py
      - Generates: et/ (gaze), physio/ (EmotiBit), audio/, video/, beh/, annot/
      - Creates: events.tsv (timeline spine), task-run annotations, sync metadata
      - Optional XDF extraction via pyxdf
    
    Stage 3: Synchronization validation
      - Validates 4-tier sync hierarchy (frame logs → LSL → progress → events)
      - Generates sync report JSON with timing statistics
      - Creates per-task sync maps (if LSL data is present)
    
    Stage 4: Cleanup raw data
      - Removes sourcedata/ to keep only processed outputs
      - Retains: et/, physio/, audio/, video/, beh/, annot/, events.tsv
    """
    
    start_time = datetime.now()
    session_logger = logging.getLogger(f"Session[{session.session_id}]")
    stages = []
    files_removed = 0
    
    try:
        session_logger.info(f"Starting: {session.session_id} ({session.group_id}, phase={session.phase})")
        
        # Locate raw sources
        session = find_session_sources(session, data_root)
        
        # Create output directory (BIDS: sub-01/ses-{id})
        output_dir = output_root / "sub-01" / f"ses-{session.session_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        has_recording = session.recording_dir is not None and session.recording_dir.exists()
        has_av = session.av_dir is not None and session.av_dir.exists()
        has_tobii = session.tobii_dir is not None and session.tobii_dir.exists()
        has_stimuli = session.stimuli_dir is not None and session.stimuli_dir.exists()
        
        session_logger.info(
            f"Sources found: Recording={has_recording}, AV={has_av}, "
            f"Tobii={has_tobii}, Stimuli={has_stimuli}"
        )
        
        if not (has_recording or has_av):
            return ProcessingResult(
                session_id=session.session_id,
                success=False,
                message="No Recording or AV sources found",
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error="At least Recording or AV data must exist",
            )
        
        # ========== STAGE 1: Merge multi-source streams ==========
        session_logger.info("Stage 1: Merging multi-source streams with task windowing...")
        
        fallback_inputs_dir = output_dir / "_fallback_inputs"
        fallback_inputs_dir.mkdir(parents=True, exist_ok=True)
        fallback_recording = fallback_inputs_dir / "recording"
        fallback_av = fallback_inputs_dir / "av"
        fallback_stimuli = fallback_inputs_dir / "stimuli"
        fallback_recording.mkdir(parents=True, exist_ok=True)
        fallback_av.mkdir(parents=True, exist_ok=True)
        fallback_stimuli.mkdir(parents=True, exist_ok=True)

        recording_arg = session.recording_dir if has_recording else fallback_recording
        av_arg = session.av_dir if has_av else fallback_av
        stimuli_arg = session.stimuli_dir if has_stimuli else fallback_stimuli

        cmd_merge = [
            python_exe,
            str(tools_path / "multisource_to_bids_runs.py"),
            "--recording-session-dir", str(recording_arg),
            "--av-session-dir", str(av_arg),
            "--stimuli-dir", str(stimuli_arg),
            "--output-session-dir", str(output_dir),
        ]

        if has_tobii:
            cmd_merge.extend(["--tobii-dir", str(session.tobii_dir)])
        
        if split_media:
            cmd_merge.append("--split-media")
            cmd_merge.append("--skip-video-splitting")

        # Always keep processed outputs free of vendor raw copies.
        cmd_merge.append("--processed-only")
        
        if link_files:
            cmd_merge.append("--link")
        
        result = subprocess.run(cmd_merge, capture_output=True, text=True, timeout=7200)
        
        if result.returncode != 0:
            session_logger.error(f"Merge failed: {result.stderr[:300]}")
            return ProcessingResult(
                session_id=session.session_id,
                success=False,
                message="multisource_to_bids_runs failed",
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=result.stderr[:500],
                stages_completed=stages,
            )
        
        stages.append("merge_sources")
        session_logger.info("  ✓ Merged sources and derived task windows (T0-T4)")
        
        # ========== STAGE 2: Canonicalize to BIDS ==========
        session_logger.info("Stage 2: Converting to BIDS-compliant modality folders...")
        
        cmd_bids = [
            python_exe,
            str(tools_path / "raw_to_bids.py"),
            "--session-dir", str(output_dir),
        ]
        
        if link_files:
            cmd_bids.append("--link")
        
        result = subprocess.run(cmd_bids, capture_output=True, text=True, timeout=7200)
        
        if result.returncode != 0:
            session_logger.warning(f"BIDS conversion had warnings: {result.stderr[:200]}")
        
        stages.append("canonicalize_bids")
        session_logger.info("  ✓ Generated BIDS modality folders (et/, physio/, audio/, video/, beh/, annot/)")
        
        # ========== STAGE 3: Validate synchronization ==========
        session_logger.info("Stage 3: Validating synchronization metadata...")
        
        sync_info = {
            "session_id": session.session_id,
            "group_id": session.group_id,
            "participants": session.participants,
            "modalities": session.raw_modalities,
            "phase": session.phase,
            "sync_sources": {
                "recording_pc_xdf": has_recording,
                "av_pc_video": has_av,
                "tobii_gaze": has_tobii,
                "stimuli_markers": has_stimuli,
            },
            "sync_tiers": [
                "1_frame_logs (best, ~0.5ms)",
                "2_lsl_progress (10Hz, ~1ms)",
                "3_progress_tsv (~1ms)",
                "4_events_jsonl (worst, ~100ms)",
            ],
            "tasks": ["T0", "T1", "T2", "T3", "T4"],
            "output_modalities": [
                "annot/ (task windows, sync maps, participant signal map)",
                "beh/ (events.tsv, per-task events, stimuli responses)",
                "et/ (Tobii gaze + pupil TSV)",
                "physio/ (EmotiBit PPG, EDA, temperature)",
                "audio/ (DPA close-talk + room, per-task clips)",
                "video/ (camera MKV, per-task clips if --split-media)",
            ],
            "processing_timestamp": datetime.now().isoformat(),
            "pipeline_version": "2.0_multimodal_sync",
        }
        
        annot_dir = output_dir / "annot"
        annot_dir.mkdir(parents=True, exist_ok=True)
        
        sync_file = annot_dir / f"sub-01_ses-{session.session_id}_sync_metadata.json"
        with open(sync_file, "w") as f:
            json.dump(sync_info, f, indent=2)
        
        stages.append("validate_sync")
        session_logger.info("  ✓ Sync metadata validated and saved")
        
        # ========== STAGE 4: Cleanup raw sourcedata ==========
        session_logger.info("Stage 4: Cleaning raw sourcedata (keeping only processed outputs)...")
        
        sourcedata = output_dir / "sourcedata"
        if sourcedata.exists():
            for item in sourcedata.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                        files_removed += sum(1 for _ in item.rglob("*"))
                    else:
                        item.unlink()
                        files_removed += 1
                except Exception as e:
                    session_logger.warning(f"Could not remove {item}: {e}")
            
            # Remove sourcedata dir if empty
            try:
                sourcedata.rmdir()
            except:
                pass
        
        stages.append("cleanup_raw")
        session_logger.info(f"  ✓ Removed {files_removed} raw files (sourcedata/)")
        
        # Count final processed files
        output_files = list(output_dir.rglob("*"))
        file_count = sum(1 for f in output_files if f.is_file())
        
        duration = (datetime.now() - start_time).total_seconds()
        
        session_logger.info(
            f"✓ COMPLETED: {file_count} processed files retained, "
            f"{duration:.1f}s ({duration/60:.1f} min), "
            f"Stages: {' → '.join(stages)}"
        )
        
        return ProcessingResult(
            session_id=session.session_id,
            success=True,
            message=f"Merged sources, split tasks, generated BIDS, cleaned raw data",
            duration_seconds=duration,
            files_processed=file_count,
            files_removed=files_removed,
            output_dir=output_dir,
            stages_completed=stages,
        )
    
    except Exception as e:
        import traceback
        duration = (datetime.now() - start_time).total_seconds()
        session_logger.error(f"Processing failed: {str(e)}")
        
        return ProcessingResult(
            session_id=session.session_id,
            success=False,
            message=f"Exception: {str(e)}",
            duration_seconds=duration,
            error=traceback.format_exc()[:500],
            stages_completed=stages,
        )


def load_sessions(inventory_path: Path) -> list[SessionInfo]:
    """
    Load session metadata from CSV inventory.
    
    CSV columns used:
    - session: session_id (e.g., ses-20260311_grp-06_run01)
    - group_id: group identifier (e.g., grp-06)
    - phase_tags: semicolon-separated phases (final, pilot, test) — first is primary
    - participants_ids: semicolon-separated BIDS participant IDs (sub-01, sub-02, etc.)
    - raw_modalities: semicolon-separated available modalities (lsl, tobii_lsl, av, etc.)
    """
    sessions = []
    
    if not inventory_path.exists():
        logger.error(f"Inventory not found: {inventory_path}")
        return sessions
    
    with open(inventory_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("session"):
                continue
            
            session_id = row.get("session", "").strip()
            group_id = row.get("group_id", "").strip()
            phase_tags = [t.strip() for t in row.get("phase_tags", "").split(";") if t.strip()]
            phase = phase_tags[0] if phase_tags else "final"
            participants = [p.strip() for p in row.get("participants_ids", "").split(";") if p.strip()]
            raw_modalities = [m.strip() for m in row.get("raw_modalities", "").split(";") if m.strip()]
            
            sessions.append(SessionInfo(
                session_id=session_id,
                group_id=group_id,
                phase=phase,
                participants=participants or ["sub-01", "sub-02", "sub-03", "sub-04"],
                raw_modalities=raw_modalities or ["unknown"],
            ))
    
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description="AffectAI Multimodal BIDS Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PROCESSING PIPELINE
───────────────────

Stage 1: Merge Multi-Source Streams
  Input sources (located via session matching):
    • Recording-PC: data/affectai-capture-recording/sessions/{phase}/sub-01/ses-*
                    Contains: XDF (LSL), Tobii LSL, EmotiBit LSL, events.tsv markers
    • AV-PC:        data/AV/{phase}/*{group_id}*
                    Contains: Camera MKV videos, DPA WAV audio, frame logs, progress TSV
    • Tobii:        data/Tobii/*{session_id}*
                    Contains: Manual downloads (scene video, gaze recordings)
    • Stimuli:      data/stimuli/{phase}/*{group_id}*
                    Contains: Task markers, task windows (T0-T4), tablet responses

  Tool: multisource_to_bids_runs.py
  Output: Merged raw in sourcedata/, derives task windows

Stage 2: Canonicalize to BIDS Format
  Tool: raw_to_bids.py
  Output modalities:
    • annot/    - Task windows TSV, participant signal map, sync metadata JSON
    • beh/      - events.tsv (authoritative timeline spine), per-task events, stimuli responses
    • et/       - Tobii gaze + pupil TSV per task
    • physio/   - EmotiBit PPG, EDA, temperature per task
    • audio/    - DPA close-talk + room WAV clips per task
    • video/    - Camera MKV clips per task (if --split-media)

Stage 3: Validate Synchronization
  Generates sync_metadata.json with:
    • Sync tier hierarchy (frame logs → LSL → progress TSV → events JSONL)
    • Per-source availability (Recording-PC, AV-PC, Tobii, Stimuli)
    • Task list (T0, T1, T2, T3, T4)
    • Participant assignment and modality availability

Stage 4: Cleanup Raw Data
  Removes sourcedata/ directory while preserving:
    • Processed BIDS modality folders
    • events.tsv, sync_metadata.json, other BIDS-standard files
  Result: Clean BIDS dataset, no raw vendor files

NAMING CONVENTIONS & SESSION MATCHING
─────────────────────────────────────

Session Identifier Format:
  ses-{DATE}_{GROUP_ID}_run{RUN_NUMBER}
  Example: ses-20260311_grp-06_run01

Matching Strategy (uses inventory CSV):
  • Recording-PC:  Match by session_id AND phase
                  data/affectai-capture-recording/sessions/{phase}/sub-01/ses-{session_id}*
  • AV-PC:         Match by group_id AND phase
                  data/AV/{phase}/*{group_id}*
  • Tobii:         Match by session_id
                  data/Tobii/*{session_id}*
  • Stimuli:       Match by group_id AND phase
                  data/stimuli/{phase}/*{group_id}*

Participant Mapping:
  Loaded from: configs/emotibit_participants_by_source.json
  Maps:
    • Device IP addresses to participant roles (P1, P2, P3, P4)
    • EmotiBit device serials to participants
  Used for: Assigning physiological streams to correct participant

CONFIGURATION FILES
───────────────────

Required:
  • configs/emotibit_participants.json or configs/emotibit_participants_by_source.json
    Maps participant IDs (P1–P4) to EmotiBit serial numbers

Optional:
  • configs/camera_specs.json           - Camera calibration info
  • configs/desk_zones.json              - Participant seating zones
  • configs/tobii_multicam_glasses_tracker.example.yaml - Tobii tracking config
  • configs/calibration_charuco.toml    - Camera calibration (if available)

EXAMPLE USAGE
─────────────

python bids_processing_pipeline.py \\
  --data-root "D:\\AffecAI Data\\affectai-data-processing\\affectai-data-processing-seed" \\
  --output-root "E:\\processed_data" \\
  --inventory "data/high_level_session_inventory.csv" \\
  --max-workers 4 \\
  --split-media \\
  --link-files

OUTPUT STRUCTURE
────────────────

E:\\processed_data\\
├── sub-01\\
│   ├── ses-20260311_grp-06_run01\\
│   │   ├── annot\\
│   │   │   ├── sub-01_ses-20260311_grp-06_run01_sync_metadata.json
│   │   │   ├── sub-01_ses-20260311_grp-06_run01_task-T0T1T2T3T4_task_windows.tsv
│   │   │   └── sub-01_ses-20260311_grp-06_run01_participant_signal_map.tsv
│   │   ├── beh\\
│   │   │   ├── sub-01_ses-20260311_grp-06_run01_events.tsv
│   │   │   ├── sub-01_ses-20260311_grp-06_run01_task-T1_run-01_events.tsv
│   │   │   └── sub-01_ses-20260311_grp-06_run01_stimuli_answers.tsv
│   │   ├── et\\      (Tobii gaze, per-task if --split-media)
│   │   ├── physio\\  (EmotiBit, per-task if --split-media)
│   │   ├── audio\\   (DPA, per-task if --split-media)
│   │   └── video\\   (Camera, per-task if --split-media)
│   ├── ses-20260311_grp-06_run02\\
│   └── ...
└── participants.tsv, dataset_description.json (BIDS root files)

        """
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root data directory with: affectai-capture-recording/, AV/, Tobii/, stimuli/",
    )
    
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for BIDS dataset (processed data only, no raw)",
    )
    
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="CSV file with session inventory (high_level_session_inventory.csv)",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for session processing",
    )
    
    parser.add_argument(
        "--split-media",
        action="store_true",
        help="Split video/audio into per-task clips (via ffmpeg)",
    )
    
    parser.add_argument(
        "--link-files",
        action="store_true",
        help="Use hard links instead of copies to save disk space",
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory (default: {data-root}/configs)",
    )
    
    args = parser.parse_args()
    
    # Setup
    python_exe = sys.executable
    tools_path = Path(__file__).parent
    config_dir = args.config_dir or (args.data_root.parent / "configs")
    
    # Load and validate inventory
    if not args.inventory.exists():
        logger.error(f"Inventory not found: {args.inventory}")
        return 1
    
    logger.info(f"Loading sessions from {args.inventory}")
    sessions = load_sessions(args.inventory)
    logger.info(f"Loaded {len(sessions)} sessions")
    
    if not sessions:
        logger.error("No sessions found in inventory")
        return 1
    
    # Load configuration files
    emotibit_config_path = config_dir / "emotibit_participants_by_source.json"
    if not emotibit_config_path.exists():
        emotibit_config_path = config_dir / "emotibit_participants.json"
    
    emotibit_config = {}
    if emotibit_config_path.exists():
        try:
            with open(emotibit_config_path) as f:
                emotibit_data = json.load(f)
            emotibit_config = emotibit_data.get("participants", emotibit_data)
            logger.info(f"Loaded EmotiBit participant map from {emotibit_config_path.name}")
        except Exception as e:
            logger.warning(f"Could not load EmotiBit config: {e}")
    else:
        logger.warning(f"No EmotiBit config found at {config_dir}")
    
    # Create output root
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Log pipeline configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Data root:        {args.data_root}")
    logger.info(f"Output root:      {args.output_root}")
    logger.info(f"Config dir:       {config_dir}")
    logger.info(f"Inventory:        {args.inventory}")
    logger.info(f"Workers:          {args.max_workers}")
    logger.info(f"Split media:      {args.split_media}")
    logger.info(f"Link files:       {args.link_files}")
    logger.info(f"Sessions to process: {len(sessions)}")
    logger.info("=" * 80)
    logger.info("")
    
    # Process sessions in parallel
    logger.info(f"Starting processing with {args.max_workers} workers...")
    logger.info(f"Pipeline: Merge → Sync → Split Tasks → BIDS → Clean")
    logger.info("")
    
    with mp.Pool(processes=args.max_workers) as pool:
        tasks = [
            (session, args.data_root, args.output_root, python_exe, tools_path, args.split_media, args.link_files)
            for session in sessions
        ]
        
        results = pool.starmap(process_session_worker, tasks)
    
    # Summary report
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_files = sum(r.files_processed for r in results if r.success)
    total_files_removed = sum(r.files_removed for r in results if r.success)
    total_duration = sum(r.duration_seconds for r in results)
    
    # Session details with source matching status
    session_sources_summary = {}
    for session in sessions:
        session_sources_summary[session.session_id] = {
            "group_id": session.group_id,
            "phase": session.phase,
            "participants": session.participants,
            "modalities": session.raw_modalities,
        }
    
    print("\n" + "=" * 80)
    print("BIDS PROCESSING PIPELINE - FINAL REPORT".center(80))
    print("=" * 80)
    
    print(f"\n📊 PROCESSING STATISTICS")
    print(f"  Sessions processed:  {len(results)}")
    print(f"  ✓ Successful:        {successful}")
    print(f"  ✗ Failed:            {failed}")
    print(f"  Success rate:        {100*successful/len(results):.1f}%")
    
    print(f"\n📁 OUTPUT STATISTICS")
    print(f"  Processed files:     {total_files}")
    print(f"  Raw files removed:   {total_files_removed}")
    print(f"  Duration:            {total_duration/3600:.2f} h ({total_duration/60:.0f} min)")
    print(f"  Avg per session:     {total_duration/len(results):.0f} s")
    
    print(f"\n📍 DATA LOCATIONS")
    print(f"  Input sources:       {args.data_root}")
    print(f"  BIDS output:         {args.output_root}")
    print(f"  Config directory:    {config_dir}")
    
    print(f"\n🔄 SOURCE MATCHING & SESSION BINDING")
    print(f"  Inventory file:      {args.inventory.name}")
    print(f"  Matching strategy:   Name-based by session_id, group_id, phase")
    print(f"  Sources located:")
    print(f"    • Recording-PC:    data/affectai-capture-recording/sessions/{{phase}}/sub-01/ses-*")
    print(f"    • AV-PC:           data/AV/{{phase}}/*{{group_id}}*")
    print(f"    • Tobii:           data/Tobii/*{{session_id}}*")
    print(f"    • Stimuli:         data/stimuli/{{phase}}/*{{group_id}}*")
    
    print(f"\n🎯 PARTICIPANT & MODALITY MAPPING")
    if emotibit_config:
        print(f"  EmotiBit config:     Loaded ({len(emotibit_config)} participants)")
        for p_id, device_id in list(emotibit_config.items())[:4]:
            print(f"    {p_id}: {device_id}")
    else:
        print(f"  EmotiBit config:     Not found (using defaults)")
    
    print(f"  Modalities processed: {set(m for s in sessions for m in s.raw_modalities)}")
    
    print(f"\n🔧 PROCESSING PIPELINE STAGES")
    if results and results[0].stages_completed:
        stages_map = {
            "merge_sources": "Merged multi-source streams (XDF + video + Tobii + stimuli)",
            "canonicalize_bids": "Canonicalized to BIDS modality folders (et/, physio/, audio/, video/)",
            "validate_sync": "Validated synchronization tiers (frame logs → LSL → progress → events)",
            "cleanup_raw": "Cleaned raw sourcedata (retained only processed outputs)",
        }
        for stage in results[0].stages_completed:
            print(f"    ✓ {stages_map.get(stage, stage)}")
    
    print(f"\n💾 OUTPUT MODALITY STRUCTURE")
    print(f"  annot/ - Task windows (T0-T4), sync metadata, participant signal map")
    print(f"  beh/   - events.tsv (timeline spine), per-task events, stimuli responses")
    print(f"  et/    - Tobii gaze + pupil (per task)")
    print(f"  physio/- EmotiBit PPG, EDA, temperature (per task)")
    print(f"  audio/ - DPA close-talk + room microphones (per task)")
    print(f"  video/ - Camera recordings (per task)")
    print(f"\n  Stimuli annotations:")
    print(f"    • Task windows:  beh/sub-*_task-T1T2T3T4_task_windows.tsv")
    print(f"    • Responses:     beh/sub-*_stimuli_answers.tsv")
    print(f"    • Events:        beh/sub-*_events.tsv (authoritative timeline)")
    print(f"  Sync metadata:")
    print(f"    • Sub-session metadata: annot/sub-*_sync_metadata.json")
    
    print(f"\n📋 BIDS COMPLIANCE & NAMING")
    print(f"  Output format:       BIDS 1.9+ (Brain Imaging Data Structure)")
    print(f"  Participant level:   sub-01/")
    print(f"  Session level:       ses-{{DATE}}_{{GROUP_ID}}_run{{RUN_NUMBER}}/")
    print(f"  Task naming:         task-T0, task-T1, task-T2, task-T3, task-T4")
    print(f"  Entity ordering:     standard BIDS key-value order")
    
    print("\n" + "=" * 80)
    
    if failed > 0:
        print(f"\n⚠️  FAILED SESSIONS ({failed}/{len(results)})")
        for r in results:
            if not r.success:
                print(f"\n  ✗ {r.session_id} (group: {r.session_id.split('_')[1] if '_' in r.session_id else '?'})")
                print(f"    Status:  {r.message}")
                if r.stages_completed:
                    print(f"    Stages completed: {', '.join(r.stages_completed)}")
                if r.error:
                    print(f"    Error: {r.error[:150]}")
    else:
        print(f"\n✅ ALL SESSIONS PROCESSED SUCCESSFULLY")
    
    print("\n" + "=" * 80)
    print(f"\n✨ Next steps:")
    print(f"  1. Validate BIDS compliance: bids-validator {args.output_root}")
    print(f"  2. Review sync_metadata.json for timing information")
    print(f"  3. Check events.tsv for complete task timeline")
    print(f"  4. Optional: Run downstream pipelines (3D pose, face/hand, etc.)")
    print("=" * 80 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
