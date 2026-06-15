"""enrich_speech_features.py

Enrich speech features with group-level context and T0-normalised relative features.

Problems with the raw speech features in dataset_15s_speech.pkl:
  1. 58% of 15s windows have zero speech (participant didn't speak in that window)
  2. Features are per-participant — no reference to how much others spoke
  3. No T0 baseline normalisation (individual speaking style differences swamp task effects)

This script adds group-relative speech features:
  speech_share              — this person's fraction of total group speaking time
  speech_share_utterances   — this person's fraction of group utterances
  group_speech_total        — sum of all 4 seats' speaking time in same window
  group_speech_entropy      — entropy of speaking distribution (high = balanced)
  group_speech_std          — std of speaking times (high = dominated by one person)
  speech_energy_rel         — this person's energy relative to group mean energy
  group_n_speakers          — count of participants who spoke in the window
  speech_t0_delta           — speech_fraction minus this person's T0 mean speech_fraction
  speech_energy_t0_delta    — speech_energy_mean minus T0 mean (baseline-corrected)

Usage
-----
  python tools/mumt/enrich_speech_features.py
  python tools/mumt/enrich_speech_features.py \\
      --input  data/mumt/dataset_15s_speech.pkl \\
      --output data/mumt/dataset_15s_speech_enriched.pkl
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SEATS = ["P1", "P2", "P3", "P4"]


def _safe_get(row: pd.Series, col: str, key: str, default: float = 0.0) -> float:
    d = row.get(col)
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return float(v)


def compute_group_speech(df: pd.DataFrame) -> pd.DataFrame:
    """Add group-level speech columns to *df* in-place.

    Groups windows by (session_id, task, window_center) — or by the nearest
    equivalent overlap key — and aggregates all participants' speech in the
    same ~15s window.
    """
    # Build a float column for per-row speech_time_sec, speech_energy_mean etc.
    df = df.copy()
    df["_st"]  = [_safe_get(r, "speech_features", "speech_time_sec")  for _, r in df.iterrows()]
    df["_sn"]  = [_safe_get(r, "speech_features", "speech_n_utterances") for _, r in df.iterrows()]
    df["_se"]  = [_safe_get(r, "speech_features", "speech_energy_mean") for _, r in df.iterrows()]
    df["_sf"]  = [_safe_get(r, "speech_features", "speech_fraction")  for _, r in df.iterrows()]

    # Group key: same session + task + approximate window position
    # Use vad_timestamp_lsl rounded to nearest 15s as window ID
    if "vad_timestamp_lsl" in df.columns:
        df["_win_key"] = (
            df["session_id"].astype(str) + "__" + df["task"].astype(str) + "__" +
            (df["vad_timestamp_lsl"] // 15).astype(int).astype(str)
        )
    else:
        # Fallback: group by session + task (coarser)
        df["_win_key"] = df["session_id"].astype(str) + "__" + df["task"].astype(str)

    # Group aggregates
    grp = df.groupby("_win_key")
    group_total = grp["_st"].transform("sum")
    group_utts  = grp["_sn"].transform("sum")
    group_mean_e = grp["_se"].transform("mean")
    group_n_sp   = grp["_st"].transform(lambda x: (x > 0).sum())
    group_std    = grp["_st"].transform("std").fillna(0.0)

    # Shannon entropy of speaking distribution per window
    def _win_entropy(vals: pd.Series) -> float:
        v = vals.values.astype(float)
        v = np.clip(v, 0, None)
        total = v.sum()
        if total < 0.01:
            return 0.0
        probs = v / total
        probs = probs[probs > 0]
        return float(scipy_entropy(probs))

    group_entropy = grp["_st"].transform(_win_entropy)

    df["_grp_total"]   = group_total.values
    df["_grp_utts"]    = group_utts.values
    df["_grp_mean_e"]  = group_mean_e.values
    df["_grp_n_sp"]    = group_n_sp.values
    df["_grp_std"]     = group_std.values
    df["_grp_entropy"] = group_entropy.values

    return df


def compute_t0_baselines(df: pd.DataFrame) -> dict[tuple, dict]:
    """Return {(session_id, seat): {feature: mean_T0_value}}."""
    t0 = df[df["task"] == "T0"]
    baselines: dict[tuple, dict] = {}
    for (ses, seat), grp in t0.groupby(["session_id", "seat"]):
        baselines[(str(ses), str(seat))] = {
            "speech_fraction":     float(grp["_sf"].mean()),
            "speech_energy_mean":  float(grp["_se"].mean()),
            "speech_time_sec":     float(grp["_st"].mean()),
        }
    return baselines


def build_enriched_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with enriched `speech_features` dicts."""
    df = compute_group_speech(df)
    t0_base = compute_t0_baselines(df)

    enriched_speech: list[dict] = []
    for _, row in df.iterrows():
        orig: dict = row.get("speech_features") or {}
        if not isinstance(orig, dict):
            orig = {}

        st  = row["_st"]
        sn  = row["_sn"]
        se  = row["_se"]
        sf  = row["_sf"]
        gt  = row["_grp_total"]
        gu  = row["_grp_utts"]
        gme = row["_grp_mean_e"]
        gns = row["_grp_n_sp"]
        gsd = row["_grp_std"]
        gent = row["_grp_entropy"]

        # Group-relative features
        speech_share       = st  / (gt  + 1e-6)          # [0,1] fraction of group speaking time
        speech_share_utts  = sn  / (gu  + 1e-6)          # fraction of group utterances
        speech_energy_rel  = se  / (gme + 1e-6)          # relative energy (1.0 = at group mean)

        # T0-delta features (baseline-corrected per participant)
        key = (str(row.get("session_id", "")), str(row.get("seat", "")))
        base = t0_base.get(key, {})
        sf_t0      = base.get("speech_fraction",    0.0)
        se_t0      = base.get("speech_energy_mean", 0.0)
        st_t0      = base.get("speech_time_sec",    0.0)
        speech_t0_delta       = sf - sf_t0
        speech_energy_t0_delta = se - se_t0
        speech_time_t0_delta  = st - st_t0

        new: dict = dict(orig)
        new.update({
            # Group-level speech dynamics
            "group_speech_total":      float(gt),
            "group_speech_entropy":    float(gent),
            "group_speech_std":        float(gsd),
            "group_n_speakers":        float(gns),
            # Relative / share features
            "speech_share":            float(speech_share),
            "speech_share_utterances": float(speech_share_utts),
            "speech_energy_rel":       float(speech_energy_rel),
            # T0-normalised deltas
            "speech_t0_delta":         float(speech_t0_delta),
            "speech_energy_t0_delta":  float(speech_energy_t0_delta),
            "speech_time_t0_delta":    float(speech_time_t0_delta),
        })
        enriched_speech.append(new)

    df = df.copy()
    df["speech_features"] = enriched_speech

    # Drop temp columns
    tmp_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=tmp_cols, inplace=True)

    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Enrich speech features with group context")
    p.add_argument("--input",  default="data/mumt/dataset_15s_speech.pkl")
    p.add_argument("--output", default="data/mumt/dataset_15s_speech_enriched.pkl")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading: %s", in_path)
    with open(in_path, "rb") as f:
        df = pickle.load(f)
    log.info("Input: %d rows", len(df))

    log.info("Enriching speech features…")
    df_out = build_enriched_features(df)

    # Report new feature keys
    sample = df_out["speech_features"].iloc[0]
    log.info("Speech feature keys after enrichment (%d total): %s",
             len(sample), sorted(sample.keys()))

    with open(out_path, "wb") as f:
        pickle.dump(df_out, f, protocol=4)
    log.info("Saved: %s  (%d rows)", out_path, len(df_out))

    # Quick correlation check
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    new_keys = ["group_speech_total", "group_speech_entropy", "group_n_speakers",
                "speech_share", "speech_share_utterances", "speech_energy_rel",
                "speech_t0_delta", "speech_energy_t0_delta"]
    for k in new_keys:
        col = [float(r.get(k, np.nan)) if isinstance(r, dict) else np.nan
               for r in df_out["speech_features"]]
        df_out[f"__{k}"] = col
    print("\nCorrelation: new speech features vs VAD:")
    for k in new_keys:
        col_name = f"__{k}"
        for dim in ["valence", "arousal", "dominance"]:
            r = df_out[[col_name, dim]].dropna().corr().iloc[0, 1]
            if abs(r) > 0.06:
                print(f"  {k:35s} vs {dim:10s}: r={r:.3f}")


if __name__ == "__main__":
    main()
