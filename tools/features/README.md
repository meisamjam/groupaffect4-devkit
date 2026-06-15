# Feature Extraction Tools

Task-aware physiology and semantic biomarker extraction for AffectAI session data.

## Scripts
- `extract_physio_features.py`: EmotiBit participant-task and rolling-window features.
- `analyze_physio_paper.py`: paper-facing physio usability, task-effect, correlation,
  and temporal-profile summaries.
- `analyze_autonomic_paper.py`: combined EmotiBit + Tobii pupil paper summaries and figures.
- `extract_pupil_features.py`: Tobii pupil participant-task and rolling-window features.
- `compute_group_dynamics.py`: dyad/group synchrony metrics from window tables.
- `build_semantic_biomarkers.py`: semantic biomarker composites from extracted features.
- `build_participant_group_comparisons.py`: participant joins with answers/annotations + group pooled comparisons.
- `run_feature_pipeline.py`: runs all scripts in sequence.
- `visualize_physio_features.py`: quick PNG summaries for physio feature/QC review.

## Typical Run
```bash
python tools/features/run_feature_pipeline.py \
  --data-root affectai-data-processing-seed/data \
  --out-dir data/derived_features \
  --window-s 30 \
  --step-s 15
```

Quick physio figures:
```bash
python tools/features/visualize_physio_features.py \
  --features-dir features \
  --out-dir figures/physio
```

Paper physio analysis:
```bash
python tools/features/analyze_physio_paper.py \
  --features-dir features \
  --results-dir results/physio \
  --figures-dir figures/physio
```

Pupil + physio paper analysis:
```bash
python tools/features/extract_pupil_features.py \
  --data-root F:/processed_data/sub-01 \
  --out-dir features \
  --window-s 30 \
  --step-s 15

python tools/features/analyze_autonomic_paper.py \
  --features-dir features \
  --results-dir results/autonomic \
  --figures-dir figures/autonomic
```

## Key Outputs
- `physio_participant_task.tsv` (canonical paper-ready EmotiBit table)
- `physio_qc_summary.tsv` (participant-task QC and missingness table)
- `physio_window_30s.tsv` (canonical rolling-window EmotiBit table)
- `physio_feature_definitions.tsv`
- `results/physio/physio_paper_feature_usability.tsv`
- `results/physio/physio_task_delta_stats.tsv`
- `results/physio/physio_session_task_summary.tsv`
- `results/physio/physio_qc_flag_counts.tsv`
- `results/physio/physio_feature_correlations.tsv`
- `results/physio/physio_temporal_profile.tsv`
- `features_pupil_participant_task.tsv`
- `features_pupil_window_30s.tsv`
- `results/autonomic/autonomic_task_delta_stats.tsv`
- `results/autonomic/autonomic_modality_coverage.tsv`
- `results/autonomic/autonomic_pupil_physio_links.tsv`
- `results/autonomic/autonomic_paper_key_findings.tsv`
- `features_physio_participant_task.tsv` (legacy alias, unless disabled)
- `features_physio_window_30s.tsv` (legacy alias, unless disabled)
- `features_pupil_participant_task.tsv`
- `features_pupil_window_30s.tsv`
- `features_group_dynamics_window_30s.tsv`
- `features_group_dynamics_task.tsv`
- `semantic_biomarkers_participant_task.tsv`
- `semantic_biomarkers_window_30s.tsv`
- `participant_features_answers_annotations.tsv`
- `group_pool_task_summary.tsv`
- `participant_vs_group_comparison.tsv`
- `biomarker_vad_label_comparison.tsv`
- `biomarker_vad_performance_by_participant.tsv`
- `biomarker_annotation_performance_by_participant.tsv`

## Notes
- Uses participant IDs `P1`-`P4` from split files (`*_acq-P*_*.tsv.gz`).
- Assumes task-split files already exist (`task-T0` to `task-T4`).
- Default channel indices can be overridden via CLI flags.
- For paper coverage tables, run physio extraction with `--include-missing-qc`.
