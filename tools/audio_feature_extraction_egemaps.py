"""Extract GeMAPSv01b (62-parameter) audio features from DPA recordings.

Processes all per-speaker DPA microphone WAV files found under --audio-root
for sessions grp-07 to grp-16 (all subdirectories are scanned automatically).

Feature set — opensmile GeMAPSv01b, 62 parameters total
  18 LLDs (smoothed with 3-frame symmetric moving average, sma3):
    Pitch        : log F0 on semitone scale from 27.5 Hz
    Jitter       : deviations in consecutive F0 period lengths          [voiced]
    F1/F2/F3 freq: formant centre frequencies                          [voiced]
    F1 bandwidth : bandwidth of the first formant                      [voiced]
    Shimmer      : peak-amplitude differences of consecutive F0 periods [voiced]
    Loudness     : perceived intensity from auditory spectrum           [all]
    HNR          : harmonics-to-noise ratio                            [voiced]
    Alpha Ratio  : energy ratio 50–1000 Hz / 1–5 kHz
    Hammarberg   : strongest peak in 0–2 kHz / strongest in 2–5 kHz
    Slope 0–500  : log-power-spectrum slope, 0–500 Hz band
    Slope 500–1500: log-power-spectrum slope, 500–1500 Hz band
    F1/F2/F3 RelE: energy of harmonic at formant relative to F0 energy [voiced]
    H1–H2        : log energy ratio of 1st to 2nd F0 harmonic          [voiced]
    H1–A3        : log energy ratio of 1st harmonic to highest in F3   [voiced]

  Functionals applied to all 18 LLDs over voiced regions (non-zero F0):
    amean, stddevNorm (coefficient of variation)  [→ 36 params]
  Additional functionals for F0 and Loudness:
    percentile 20/50/80, percentile range 20–80,
    mean/std of rising slope, mean/std of falling slope [→ 16 params = 52 total]
  Unvoiced-region means for Alpha Ratio, Hammarberg, Slope 0–500, Slope 500–1500
    [→ 4 params = 56 total]
  Prosodic descriptors:
    loudnessPeaksPerSec, VoicedSegmentsPerSec,
    MeanVoicedSegmentLengthSec, StddevVoicedSegmentLengthSec,
    MeanUnvoicedSegmentLength, StddevUnvoicedSegmentLength
    [→ 6 params = 62 total]

Analysis configuration (GeMAPSv01b defaults):
  Sampling rate   : 16 kHz (audio resampled if needed)
  Frame size      : 60 ms (960 samples)
  Hop size        : 10 ms (160 samples)
  Window function : Hamming
  Pre-emphasis    : disabled
  Frequency range : 0–8 kHz (Nyquist)
  F0 search range : 50–500 Hz  [note: spec requested 55–1000 Hz; GeMAPSv01b
                                default is 50–500 Hz, which cannot be changed
                                without a custom SMILExtract config]
  Silence threshold: -60 dB for VAD

Mic → participant mapping (consistent across all groups):
  mic9  → P1 
  mic10 → P2 
  mic11 → P3 
  mic12 → P4

Input structure expected under --audio-root:
  <audio-root>/
    ses-YYYYMMDD_grp-NN_runMM/
      sub-01_ses-..._task-T2_run-01_acq-dpa_micX_aud.wav
      ...  (mic9, mic10, mic11, mic12)

Outputs (written to --out-dir):
  t2_geMAPS_features_all.csv           — combined table: all sessions × speakers

Visualisation outputs (written to --figures-dir, default <out-dir>/figures/):
  radar_group_acoustics.png            — spider/radar chart of 6 key features per group
  pca_speakers.png                     — PCA scatter of all speakers coloured by group
  summary_table.csv                    — mean ± SD per group for all 62 features
  (pass --skip-visualise to suppress)

Usage:
    python tools/audio_feature_extraction_egemaps.py \\
        --audio-root "C:/path/to/05_audio data/ses-...-3-001" \\
        --out-dir "audio analysis/egemaps"

    # Restrict to specific groups:
    python tools/audio_feature_extraction_egemaps.py \\
        --audio-root "..." --groups grp-07 grp-08 grp-15

    # Extract only, skip figures:
    python tools/audio_feature_extraction_egemaps.py \\
        --audio-root "..." --skip-visualise
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import opensmile
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Mic number → anonymous seat/participant label (consistent across all sessions)
MIC_TO_PARTICIPANT: dict[str, str] = {
    "mic9":  "P1",
    "mic10": "P2",
    "mic11": "P3",
    "mic12": "P4",
}

# Regex to parse WAV filename:
# sub-01_ses-20260312_grp-07_run01_task-T2_run-01_acq-dpa_mic9_aud.wav
_WAV_RE = re.compile(
    r"(?:sub-(?P<sub>[^_]+)_)?"                      # sub-01 (optional)
    r"ses-(?P<ses_date>\d{8})_"                       # date
    r"(?P<grp>grp-\d+)_"                              # group id
    r"run(?P<ses_run>\d+)"                             # session run number
    r".*?"                                             # may include task/run entities
    r"_acq-dpa_(?P<mic>mic\d+)_aud\.wav$",
    re.IGNORECASE,
)


# ── opensmile extractor ────────────────────────────────────────────────────────

def build_smile() -> opensmile.Smile:
    """Create a GeMAPSv01b (62-functional) opensmile extractor."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


# ── WAV discovery ──────────────────────────────────────────────────────────────

def discover_wavs(audio_root: Path, groups: list[str] | None = None) -> list[Path]:
    """Return all T2 DPA base-WAV files under audio_root.

    Excludes _hp, _hp_cal, _hp_norm variants and anything inside a 'peaks'
    subdirectory. Optionally filters to a list of group IDs (e.g. ['grp-07']).
    """
    # Match only the canonical _aud.wav (no extra suffixes before .wav)
    all_wavs = sorted(audio_root.rglob("*_acq-dpa_mic*_aud.wav"))

    # Exclude processed variants and Reaper peak caches
    canonical: list[Path] = []
    for wav in all_wavs:
        name = wav.name.lower()
        if "peaks" in wav.parts:
            continue
        # Reject variants: _aud_hp.wav, _aud_hp_cal.wav, _aud_hp_norm.wav
        if re.search(r"_aud_hp", name):
            continue
        canonical.append(wav)

    if groups:
        group_set = {g.lower() for g in groups}
        canonical = [w for w in canonical if any(g in w.name.lower() for g in group_set)]

    return canonical


# ── Single-file extraction ─────────────────────────────────────────────────────

def extract_one(smile: opensmile.Smile, wav_path: Path) -> dict | None:
    """Extract 62 GeMAPSv01b features from one WAV file.

    Returns a flat dict of metadata + feature values, or None on failure.
    Privacy: participant IDs are P1–P4 only; no real names are stored.
    """
    m = _WAV_RE.search(wav_path.name)
    if m is None:
        log.warning("Filename does not match expected pattern — skipping: %s", wav_path.name)
        return None

    group_id = m.group("grp")              # e.g. "grp-07"
    ses_date = m.group("ses_date")         # e.g. "20260312"
    ses_run  = m.group("ses_run")          # e.g. "01"
    mic      = m.group("mic")             # e.g. "mic9"
    session_id = f"{ses_date}_{group_id}_run{ses_run}"   # e.g. "20260312_grp-07_run01"
    participant_id = MIC_TO_PARTICIPANT.get(mic, mic)    # P1–P4

    log.info("  [%s | %s | %s]  %s", group_id, session_id, participant_id, wav_path.name)

    try:
        features_df = smile.process_file(str(wav_path))
    except Exception as exc:
        log.error("opensmile failed on %s: %s", wav_path.name, exc)
        return None

    if features_df.empty:
        log.warning("No features returned for %s", wav_path.name)
        return None

    row: dict = {
        "session_id":     session_id,
        "group_id":       group_id,
        "participant_id": participant_id,
        "mic":            mic,
        "wav_file":       wav_path.name,
    }
    row.update(features_df.iloc[0].to_dict())
    return row


# ── Visualisation ─────────────────────────────────────────────────────────────

# 6 features chosen for the radar chart (interpretable, diverse dimensions)
RADAR_FEATURES: list[tuple[str, str]] = [
    ("F0semitoneFrom27.5Hz_sma3nz_amean",       "Pitch (F0)"),
    ("loudness_sma3_amean",                      "Loudness"),
    ("VoicedSegmentsPerSec",                     "Speech rate"),
    ("HNRdBACF_sma3nz_amean",                   "HNR (clarity)"),
    ("jitterLocal_sma3nz_amean",                 "Jitter"),
    ("F0semitoneFrom27.5Hz_sma3nz_stddevNorm",  "F0 variability"),
]

GROUP_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _normalise_01(series: "pd.Series") -> "pd.Series":
    """Min-max normalise a series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        import numpy as np
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def plot_radar(df: pd.DataFrame, out_path: Path) -> None:
    """Spider/radar chart: one polygon per group, 6 acoustic dimensions."""
    import numpy as np
    import matplotlib.pyplot as plt

    feat_cols   = [f for f, _ in RADAR_FEATURES]
    feat_labels = [lbl for _, lbl in RADAR_FEATURES]
    n_feat = len(feat_cols)

    grp_means = df.groupby("group_id")[feat_cols].mean()
    for col in feat_cols:
        grp_means[col] = _normalise_01(grp_means[col])

    groups = grp_means.index.tolist()
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, grp in enumerate(groups):
        values = grp_means.loc[grp, feat_cols].tolist()
        values += values[:1]
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        ax.plot(angles, values, linewidth=2, color=color, label=grp)
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, title="Group", title_fontsize=10)
    ax.set_title("Group acoustic profiles — T2 Mini-Negotiation task\n"
                 "(min-max normalised across groups)",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved radar chart: %s", out_path)


def plot_pca(df: pd.DataFrame, out_path: Path) -> None:
    """PCA scatter of all speakers on PC1 × PC2, coloured by group."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    meta_cols = {"session_id", "group_id", "participant_id", "mic", "wav_file"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X_scaled = StandardScaler().fit_transform(df[feat_cols].values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var1, var2 = pca.explained_variance_ratio_ * 100

    groups = df["group_id"].tolist()
    unique_groups = sorted(set(groups))
    color_map = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (x, y) in enumerate(coords):
        grp = groups[i]
        pid = df.iloc[i]["participant_id"]
        color = color_map[grp]
        ax.scatter(x, y, color=color, s=100, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(pid, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", color=color)

    handles = [mpatches.Patch(color=color_map[g], label=g) for g in unique_groups]
    ax.legend(handles=handles, title="Group", title_fontsize=10,
              fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.set_title("PCA of 62 GeMAPSv01b features — all speakers (T2 task)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved PCA scatter: %s", out_path)
    log.info("  PC1 explains %.1f%%, PC2 explains %.1f%% of variance", var1, var2)


def write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """CSV with mean ± SD per group for all 62 features."""
    meta_cols = {"session_id", "group_id", "participant_id", "mic", "wav_file"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    rows = []
    for grp, gdf in df.groupby("group_id"):
        row: dict = {"group_id": grp, "n_speakers": len(gdf)}
        for col in feat_cols:
            row[f"{col}_mean"] = round(gdf[col].mean(), 5)
            row[f"{col}_sd"]   = round(gdf[col].std(), 5)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    log.info("Saved summary table: %s  (%d groups × %d columns)",
             out_path, len(summary), len(summary.columns))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--audio-root", type=Path, required=True,
        help=(
            "Root directory containing per-session subdirectories "
            "(ses-YYYYMMDD_grp-NN_runMM/) with T2 DPA WAV files. "
            "All matching subdirectories are processed automatically."
        ),
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("audio analysis/egemaps"),
        help="Output directory for CSV files. Created if absent. "
             "Default: 'audio analysis/egemaps'",
    )
    ap.add_argument(
        "--groups", nargs="+", default=None, metavar="GRP",
        help="Restrict processing to specific group IDs, "
             "e.g. --groups grp-07 grp-15. Processes all groups if omitted.",
    )
    ap.add_argument(
        "--skip-visualise", action="store_true",
        help="Skip figure and summary-table generation after extraction.",
    )
    ap.add_argument(
        "--figures-dir", type=Path, default=None, metavar="DIR",
        help="Output directory for visualisations. "
             "Defaults to <out-dir>/figures.",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {args.audio_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover WAV files ────────────────────────────────────────────────────
    wavs = discover_wavs(args.audio_root, groups=args.groups)
    if not wavs:
        raise SystemExit(
            f"No T2 DPA base WAV files found under {args.audio_root}. "
            "Check --audio-root and confirm files match the pattern "
            "*_acq-dpa_mic*_aud.wav (without _hp/_cal suffixes)."
        )

    group_filter_msg = f" (groups: {args.groups})" if args.groups else ""
    log.info("Found %d WAV files to process%s", len(wavs), group_filter_msg)

    # ── Build extractor ───────────────────────────────────────────────────────
    smile = build_smile()
    log.info(
        "opensmile GeMAPSv01b — %d features per file",
        len(smile.feature_names),
    )

    # ── Extract features ──────────────────────────────────────────────────────
    rows: list[dict] = []
    failed: list[str] = []

    for wav_path in wavs:
        row = extract_one(smile, wav_path)
        if row is not None:
            rows.append(row)
        else:
            failed.append(wav_path.name)

    if not rows:
        raise SystemExit("No features extracted — check audio root and file naming.")

    if failed:
        log.warning("%d file(s) failed extraction: %s", len(failed), failed)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Ensure consistent column order: metadata first, then 62 feature columns
    meta_cols = ["session_id", "group_id", "participant_id", "mic", "wav_file"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    # Sort by session then participant for readability
    df = df.sort_values(["group_id", "participant_id"]).reset_index(drop=True)

    # ── Write combined CSV ────────────────────────────────────────────────────
    out_csv = args.out_dir / "t2_geMAPS_features_all.csv"
    df.to_csv(out_csv, index=False)
    log.info(
        "Saved combined table: %d rows × %d feature columns → %s",
        len(df), len(feature_cols), out_csv,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f"{'Session':<28} {'Participant':>12} {'F0 amean':>10} {'Loudness amean':>16}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(
            f"{r['session_id']:<28} {r['participant_id']:>12} "
            f"{r.get('F0semitoneFrom27.5Hz_sma3nz_amean', float('nan')):>10.2f} "
            f"{r.get('loudness_sma3_amean', float('nan')):>16.4f}"
        )
    print()
    print(f"Total: {len(df)} speaker-session rows,  {len(feature_cols)} features each.")
    print(f"Output: {out_csv}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    if not args.skip_visualise:
        fig_dir = args.figures_dir if args.figures_dir else args.out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_radar(df, fig_dir / "radar_group_acoustics.png")
        plot_pca(df,   fig_dir / "pca_speakers.png")
        write_summary_table(df, fig_dir / "summary_table.csv")
        print()
        print("Visualisation files:")
        for f in sorted(fig_dir.iterdir()):
            print(f"  {f}")


if __name__ == "__main__":
    main()
