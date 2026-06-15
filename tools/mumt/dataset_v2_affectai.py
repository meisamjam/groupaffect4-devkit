"""dataset_v2_affectai.py

Extended PretrainDataset for the dual-stream MuMTAffect v2 architecture.

Adds two new self-supervised targets beyond the original PretrainDataset:
  - delta_summary: difference between next-window and current-window summary features
  - recon_target + recon_mask: masked modality reconstruction targets

Each sample returns a 18-tuple:
  (gaze_seq, pupil_seq, eda_seq, ppg_seq, imu_seq,
   summary, subject_idx, session_idx, task_idx, personality,
   sex_label, age, next_summary, has_next,
   delta_summary, has_delta, recon_target, recon_mask)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_affectai import (
    BIG_FIVE_COLS,
    EDA_SEQ_COLS,
    GAZE_SEQ_COLS,
    IMU_SEQ_COLS,
    PPG_SEQ_COLS,
    PUPIL_SEQ_COLS,
    _TASK_TO_IDX,
    flatten_features,
    make_session2idx,
    make_user2idx,
    noise_injection,
    seq_to_array,
    time_warp,
)

# Number of features per modality used as reconstruction target dimension
# These are the mean-pooled modality encoder output dimensions (d_model_enc=64)
RECON_D_PER_MODALITY = 64
N_MODALITIES = 5
MASK_PROB = 0.15  # probability of masking each modality per sample


class PretrainDatasetV2(Dataset):
    """Dataset for Phase 0 dual-stream pretraining with extended objectives.

    Beyond the original PretrainDataset targets, adds:
      - delta_summary: normalized (summary_t+1 - summary_t) for temporal change prediction
      - recon_target: per-modality segment means as reconstruction targets
      - recon_mask: random binary mask indicating which modalities to reconstruct
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict[str, int],
        session2idx: dict[str, int],
        summary_key_order: list[str],
        device: torch.device | None = None,
        augment: bool = False,
        participants_tsv: str | None = None,
        scalers: dict | None = None,
        normalize_targets: bool = True,
        mask_prob: float = MASK_PROB,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.session2idx = session2idx
        self.summary_key_order = summary_key_order
        self.device = device if device is not None else torch.device("cpu")
        self.augment = augment
        self.scalers = scalers or {}
        self.normalize_targets = normalize_targets
        self.mask_prob = mask_prob

        # Eagerly materialise sequences
        print(f"PretrainDatasetV2: materialising {len(self.df)} sequences …", flush=True)
        self._gaze = [seq_to_array(r["gaze_seq"], GAZE_SEQ_COLS) for _, r in self.df.iterrows()]
        self._pupil = [seq_to_array(r["pupil_seq"], PUPIL_SEQ_COLS) for _, r in self.df.iterrows()]
        self._eda = [seq_to_array(r["eda_seq"], EDA_SEQ_COLS) for _, r in self.df.iterrows()]
        self._ppg = [seq_to_array(r["ppg_seq"], PPG_SEQ_COLS) for _, r in self.df.iterrows()]
        self._imu = [seq_to_array(r["imu_seq"], IMU_SEQ_COLS) for _, r in self.df.iterrows()]

        # Apply scalers
        if self.scalers:
            for i in range(len(self.df)):
                if "gaze" in self.scalers:
                    self._gaze[i] = self.scalers["gaze"].transform(self._gaze[i])
                if "pupil" in self.scalers:
                    self._pupil[i] = self.scalers["pupil"].transform(self._pupil[i])
                if "eda" in self.scalers:
                    self._eda[i] = self.scalers["eda"].transform(self._eda[i])
                if "ppg" in self.scalers:
                    self._ppg[i] = self.scalers["ppg"].transform(self._ppg[i])
                if "imu" in self.scalers:
                    self._imu[i] = self.scalers["imu"].transform(self._imu[i])
            self._gaze = [np.clip(a, -10, 10) for a in self._gaze]
            self._pupil = [np.clip(a, -10, 10) for a in self._pupil]
            self._eda = [np.clip(a, -10, 10) for a in self._eda]
            self._ppg = [np.clip(a, -10, 10) for a in self._ppg]
            self._imu = [np.clip(a, -10, 10) for a in self._imu]

        # Summary features
        self._summary: list[np.ndarray] = []
        for _, r in self.df.iterrows():
            all_feats: dict = {}
            for col in ["gaze_features", "pupil_features", "eda_features",
                        "ppg_features", "imu_features"]:
                fd = r.get(col, {})
                if isinstance(fd, dict):
                    all_feats.update(fd)
            self._summary.append(flatten_features(all_feats, key_order=self.summary_key_order))

        # Build reconstruction targets: per-modality segment means
        # Each modality array is (T, F_mod). We take the mean across T as a dense target.
        # Then zero-pad to RECON_D_PER_MODALITY for uniform shape.
        self._recon_targets: list[np.ndarray] = []
        for i in range(len(self.df)):
            modality_means = []
            for arr in [self._gaze[i], self._pupil[i], self._eda[i],
                        self._ppg[i], self._imu[i]]:
                m = arr.mean(axis=0).astype(np.float32)  # (F_mod,)
                # Pad to RECON_D_PER_MODALITY
                padded = np.zeros(RECON_D_PER_MODALITY, dtype=np.float32)
                padded[:len(m)] = m
                modality_means.append(padded)
            self._recon_targets.append(
                np.stack(modality_means, axis=0)  # (N_MODALITIES, RECON_D_PER_MODALITY)
            )

        # Build next-window index
        self._next_idx = np.full(len(self.df), -1, dtype=np.int64)
        key_to_idx: dict[tuple[str, str, str, int], int] = {}
        for i, r in self.df.iterrows():
            key = (
                str(r.get("session_id", "")),
                str(r.get("seat", "")),
                str(r.get("task", "T0")),
                int(r.get("window_index", -1)),
            )
            key_to_idx[key] = int(i)
        for i, r in self.df.iterrows():
            next_key = (
                str(r.get("session_id", "")),
                str(r.get("seat", "")),
                str(r.get("task", "T0")),
                int(r.get("window_index", -1)) + 1,
            )
            self._next_idx[i] = key_to_idx.get(next_key, -1)

        print("PretrainDatasetV2: materialisation complete.", flush=True)

        # Build demographics lookup
        self._demo: dict[tuple[str, str], dict] = {}
        if participants_tsv is not None:
            try:
                pts = pd.read_csv(participants_tsv, sep="\t")
                for _, row in pts.iterrows():
                    raw_ses = str(row.get("session_id", "")).strip()
                    seat = str(row.get("seat", "")).strip()
                    ses_key = raw_ses.lstrip("ses-")
                    sex_raw = str(row.get("sex", "unknown")).lower()
                    sex_int = 0 if sex_raw == "female" else (1 if sex_raw == "male" else -1)
                    age_val = float(row.get("age", float("nan")))
                    self._demo[(ses_key, seat)] = {"sex": sex_int, "age": age_val}
            except (FileNotFoundError, ValueError):
                pass

        # Normalisation statistics
        self._personality_mean = np.zeros(len(BIG_FIVE_COLS), dtype=np.float32)
        self._personality_std = np.ones(len(BIG_FIVE_COLS), dtype=np.float32)
        self._age_mean = 0.0
        self._age_std = 1.0
        self._summary_mean = np.zeros(len(self.summary_key_order), dtype=np.float32)
        self._summary_std = np.ones(len(self.summary_key_order), dtype=np.float32)

        if self.normalize_targets:
            per_vals = self.df[BIG_FIVE_COLS].to_numpy(dtype=np.float32)
            per_vals = np.nan_to_num(per_vals, nan=0.0)
            self._personality_mean = per_vals.mean(axis=0).astype(np.float32)
            self._personality_std = np.clip(per_vals.std(axis=0), 1e-6, None).astype(np.float32)

            age_vals: list[float] = []
            for _, row in self.df.iterrows():
                ses_key = str(row.get("session_id", "")).lstrip("ses-")
                seat = str(row.get("seat", ""))
                demo = self._demo.get((ses_key, seat), {})
                a = float(demo.get("age", float("nan")))
                if not np.isnan(a):
                    age_vals.append(a)
            if age_vals:
                age_arr = np.array(age_vals, dtype=np.float32)
                self._age_mean = float(age_arr.mean())
                self._age_std = float(max(age_arr.std(), 1e-6))

            if self._summary:
                summary_stack = np.vstack(self._summary).astype(np.float32)
                self._summary_mean = summary_stack.mean(axis=0).astype(np.float32)
                self._summary_std = np.clip(summary_stack.std(axis=0), 1e-6, None).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dev = self.device

        gaze_seq = self._gaze[idx].copy()
        pupil_seq = self._pupil[idx].copy()
        eda_seq = self._eda[idx].copy()
        ppg_seq = self._ppg[idx].copy()
        imu_seq = self._imu[idx].copy()
        summary = self._summary[idx]

        if self.augment:
            orig_len = gaze_seq.shape[0]
            from scipy.signal import resample as _resample
            gaze_seq = _resample(noise_injection(time_warp(gaze_seq)), orig_len, axis=0).astype(np.float32)
            pupil_seq = _resample(noise_injection(time_warp(pupil_seq)), orig_len, axis=0).astype(np.float32)
            eda_seq = _resample(noise_injection(time_warp(eda_seq)), orig_len, axis=0).astype(np.float32)
            ppg_seq = _resample(noise_injection(time_warp(ppg_seq)), orig_len, axis=0).astype(np.float32)
            imu_seq = _resample(noise_injection(time_warp(imu_seq)), orig_len, axis=0).astype(np.float32)

        # Labels
        subject_idx = self.user2idx.get(str(row["subject_id"]), 0)
        session_idx = self.session2idx.get(str(row["session_id"]), 0)
        task_idx = _TASK_TO_IDX.get(str(row.get("task", "T0")).upper(), 0)

        personality = np.array(
            [row.get(c, 0.0) for c in BIG_FIVE_COLS], dtype=np.float32
        )
        personality = np.nan_to_num(personality, nan=0.0)
        if self.normalize_targets:
            personality = (personality - self._personality_mean) / self._personality_std

        # Demographics
        ses_key = str(row.get("session_id", "")).lstrip("ses-")
        seat = str(row.get("seat", ""))
        demo = self._demo.get((ses_key, seat), {})
        sex_int = int(demo.get("sex", -1))
        age_val = float(demo.get("age", 0.0))
        if np.isnan(age_val):
            age_val = 0.0
        if self.normalize_targets:
            age_val = (age_val - self._age_mean) / self._age_std

        # Next-summary and delta
        next_i = int(self._next_idx[idx])
        if next_i >= 0:
            next_summary = self._summary[next_i].copy()
            has_next = 1.0
            # Delta = next - current (raw, before normalization of next)
            delta_summary = (next_summary - summary).astype(np.float32)
            has_delta = 1.0
        else:
            next_summary = np.zeros_like(summary, dtype=np.float32)
            has_next = 0.0
            delta_summary = np.zeros_like(summary, dtype=np.float32)
            has_delta = 0.0

        if self.normalize_targets:
            next_summary = (next_summary - self._summary_mean) / self._summary_std
            # Delta is already a difference; normalize by summary std for scale consistency
            if has_delta > 0.5:
                delta_summary = delta_summary / self._summary_std

        # Masked reconstruction targets
        recon_target = self._recon_targets[idx].copy()  # (N_MOD, D_RECON)
        # Random mask: each modality masked independently with mask_prob
        recon_mask = (np.random.random(N_MODALITIES) < self.mask_prob).astype(np.float32)
        # Ensure at least one is masked (for non-trivial loss)
        if recon_mask.sum() == 0:
            recon_mask[np.random.randint(N_MODALITIES)] = 1.0

        return (
            torch.tensor(gaze_seq, device=dev),                                # 0
            torch.tensor(pupil_seq, device=dev),                               # 1
            torch.tensor(eda_seq, device=dev),                                 # 2
            torch.tensor(ppg_seq, device=dev),                                 # 3
            torch.tensor(imu_seq, device=dev),                                 # 4
            torch.tensor(summary, device=dev),                                 # 5
            torch.tensor(subject_idx, dtype=torch.long, device=dev),           # 6
            torch.tensor(session_idx, dtype=torch.long, device=dev),           # 7
            torch.tensor(task_idx, dtype=torch.long, device=dev),              # 8
            torch.tensor(personality, device=dev),                              # 9
            torch.tensor(sex_int, dtype=torch.long, device=dev),               # 10
            torch.tensor(age_val, dtype=torch.float32, device=dev),            # 11
            torch.tensor(next_summary, dtype=torch.float32, device=dev),       # 12
            torch.tensor(has_next, dtype=torch.float32, device=dev),           # 13
            torch.tensor(delta_summary, dtype=torch.float32, device=dev),      # 14
            torch.tensor(has_delta, dtype=torch.float32, device=dev),          # 15
            torch.tensor(recon_target, dtype=torch.float32, device=dev),       # 16 (N_MOD, D_RECON)
            torch.tensor(recon_mask, dtype=torch.float32, device=dev),         # 17 (N_MOD,)
        )
