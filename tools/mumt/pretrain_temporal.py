"""pretrain_temporal.py

Self-supervised contrastive pre-training of per-modality temporal encoders
using physiological sequences from the augmented pool (8221 windows, T=400).

Why this matters
----------------
At N=103 labeled windows, neural temporal encoders (Conv1D, GRU) cannot learn
what physiological signals look like from scratch — they must simultaneously
learn signal representations AND VAD mappings with far too few examples.

SVM sidesteps this by using 49 hand-engineered summary statistics (mean, std,
percentiles) that already encode temporal information.  The encoder must
rediscover these statistics (and ideally better ones) from raw data.

Solution: pre-train encoders on 8221 unlabeled pool sequences using SimCLR-
style contrastive learning, then fine-tune on 103 labeled windows.  The
pre-training step gives encoders a physiologically-meaningful initialization;
fine-tuning then only needs to learn the label mapping.

Architecture
------------
  Pre-training: per-modality encoder → projection head → NT-Xent loss
    Two augmented views of each window are created with augment_sequence().
    The NT-Xent loss pulls representations of the same window together and
    pushes different windows apart.

  Fine-tuning: load pre-trained encoder weights → TemporalFusionNet trunk
    The projection head is discarded; the trunk + heads are trained on
    103 labeled windows, optionally with seq-aug and AP1.

Usage
-----
  # Pre-train + fine-tune (default)
  python tools/mumt/pretrain_temporal.py

  # Pre-train only, save weights
  python tools/mumt/pretrain_temporal.py --pretrain-only --save-weights results/pretrain_conv1d.pt

  # Fine-tune from saved pre-trained weights
  python tools/mumt/pretrain_temporal.py --load-weights results/pretrain_conv1d.pt --no-pretrain

  # Compare pre-trained vs random init
  python tools/mumt/pretrain_temporal.py --encoder conv1d --compare-random
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset_affectai import (  # noqa: E402
    BIG_FIVE_COLS,
    EDA_SEQ_COLS,
    GAZE_SEQ_COLS,
    IMU_SEQ_COLS,
    PPG_SEQ_COLS,
    PUPIL_SEQ_COLS,
    build_summary_key_order,
    flatten_features,
    seq_to_array,
)
from model_temporal import (  # noqa: E402
    MODALITIES,
    MODALITY_DIMS,
    SUMMARY_DIM,
    Conv1DEncoder,
    GRUEncoder,
    SoftVADLoss,
    TemporalFusionNet,
)
from train_simple import (  # noqa: E402
    bin_vad_from_thresholds,
    compute_tertile_thresholds,
    task_split,
)
from train_temporal import (  # noqa: E402
    BFI_COLS,
    FEAT_COLS,
    MODALITY_COLS,
    VAD_DIMS,
    LabeledDataset,
    SequenceScaler,
    SummaryScaler,
    augment_sequence,
    build_pool_pseudo_labels,
    collate_labeled,
    compute_bfi_similarity_map,
    extract_summary,
    fit_seq_scalers,
    get_hard_labels,
    make_one_hot_soft,
    compute_class_weight_tensors,
)


# ── Contrastive pre-training components ───────────────────────────────────────

class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for SimCLR contrastive loss."""

    def __init__(self, input_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent contrastive loss (SimCLR).

    Parameters
    ----------
    z1, z2 : (B, D) L2-normalised embeddings of two augmented views.
    temperature : softmax temperature (lower = harder negatives).
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)          # (2B, D)
    sim = torch.mm(z, z.T) / temperature    # (2B, 2B)
    # Mask self-similarity on diagonal
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    # Positive pairs: (z1[i], z2[i]) — z2[i] is at index i+B and vice versa
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


class PoolSequenceDataset(Dataset):
    """Pre-training dataset: returns two augmented views per pool window.

    Sequences are pre-computed at init time to avoid repeated DataFrame access.
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        seq_scalers: dict[str, SequenceScaler],
    ) -> None:
        log.info("Pre-computing pool sequences for pre-training (%d windows)…", len(pool))
        # Pre-compute all sequences as numpy arrays: {mod: (N, T, D)}
        self.seq_arrays: dict[str, np.ndarray] = {}
        for mod, cols in MODALITY_COLS.items():
            arrays = []
            for _, row in pool.iterrows():
                arr = seq_to_array(row[f"{mod}_seq"], cols)
                arr = seq_scalers[mod].transform(arr)
                arrays.append(arr)
            self.seq_arrays[mod] = np.stack(arrays, axis=0)
        self.N = len(pool)
        log.info("Pool sequences pre-computed: %s", {m: v.shape for m, v in self.seq_arrays.items()})

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor]]:
        # Two independently augmented views of the same window
        views = []
        for _ in range(2):
            seqs = {}
            for mod in MODALITIES:
                arr = self.seq_arrays[mod][idx].copy()
                arr = augment_sequence(arr)
                seqs[mod] = torch.from_numpy(arr)
            views.append(seqs)
        return {"view1": views[0], "view2": views[1]}


def collate_two_views(batch: list[dict]) -> dict:
    result: dict = {"view1": {}, "view2": {}}
    for view_key in ("view1", "view2"):
        for mod in MODALITIES:
            result[view_key][mod] = torch.stack([b[view_key][mod] for b in batch])
    return result


# ── Pre-training loop ──────────────────────────────────────────────────────────

def get_temporal_embedding(
    encoders: nn.ModuleDict,
    sequences: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run all modality encoders and concatenate outputs: (B, temporal_total)."""
    parts = [encoders[m](sequences[m]) for m in MODALITIES]
    return torch.cat(parts, dim=-1)


def pretrain_encoders(
    pool: pd.DataFrame,
    seq_scalers: dict[str, SequenceScaler],
    encoder_type: str,
    enc_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    device: torch.device,
) -> dict[str, nn.Module]:
    """Pre-train per-modality encoders with NT-Xent contrastive loss.

    Returns a ModuleDict of trained encoders (same structure as TemporalFusionNet.encoders).
    """
    # Build encoders
    encoders = nn.ModuleDict()
    for name, d_m in MODALITY_DIMS.items():
        if encoder_type == "conv1d":
            encoders[name] = Conv1DEncoder(d_m, enc_dim)
        else:
            encoders[name] = GRUEncoder(d_m, hidden_size=enc_dim)
    encoders = encoders.to(device)

    # Projection head: input is temporal_total
    if encoder_type == "gru":
        temporal_total = len(MODALITIES) * 2 * enc_dim
    else:
        temporal_total = len(MODALITIES) * enc_dim
    proj_head = ProjectionHead(temporal_total, proj_dim=128).to(device)

    ds = PoolSequenceDataset(pool, seq_scalers)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_two_views, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        list(encoders.parameters()) + list(proj_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info("Pre-training %s encoders for %d epochs (N=%d, temp=%.2f)",
             encoder_type, epochs, len(ds), temperature)

    for epoch in range(epochs):
        encoders.train()
        proj_head.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            optimizer.zero_grad()
            v1 = {k: v.to(device) for k, v in batch["view1"].items()}
            v2 = {k: v.to(device) for k, v in batch["view2"].items()}
            emb1 = get_temporal_embedding(encoders, v1)
            emb2 = get_temporal_embedding(encoders, v2)
            z1 = proj_head(emb1)
            z2 = proj_head(emb2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoders.parameters()) + list(proj_head.parameters()), 1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            log.info("  Pretrain epoch %d/%d  loss=%.4f", epoch + 1, epochs, epoch_loss / max(n_batches, 1))

    log.info("Pre-training complete.")
    return encoders


# ── Fine-tuning with pre-trained encoders ─────────────────────────────────────

def finetune(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_order: list[str],
    encoder_type: str,
    enc_dim: int,
    dropout: float,
    pretrained_encoders: dict[str, nn.Module] | None,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seq_aug: bool,
    device: torch.device,
    freeze_epochs: int = 10,
) -> dict:
    """Fine-tune TemporalFusionNet from pre-trained (or random) encoder weights.

    Parameters
    ----------
    freeze_epochs : train trunk+heads with frozen encoders for this many epochs
                    before unfreezing all parameters.
    """
    thresholds = compute_tertile_thresholds(train_df)
    seq_scalers = fit_seq_scalers(train_df)

    train_summary = extract_summary(train_df, key_order)
    summary_sc = SummaryScaler().fit(train_summary)

    def make_loader(df: pd.DataFrame, shuffle: bool = False, augment: bool = False) -> DataLoader:
        ds = LabeledDataset(df, key_order, thresholds, seq_scalers,
                            use_sequences=True, augment=augment)
        ds.summary = summary_sc.transform(ds.summary)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_labeled, drop_last=False)

    train_loader = make_loader(train_df, shuffle=True, augment=seq_aug)
    val_loader   = make_loader(val_df,   shuffle=False)
    test_loader  = make_loader(test_df,  shuffle=False)

    bfi_dim = len([c for c in BFI_COLS if c in train_df.columns])
    model = TemporalFusionNet(
        encoder_type=encoder_type,
        enc_dim=enc_dim,
        dropout=dropout,
        bfi_dim=bfi_dim,
    ).to(device)

    # Load pre-trained encoder weights if available
    if pretrained_encoders is not None:
        for name in MODALITIES:
            model.encoders[name].load_state_dict(pretrained_encoders[name].state_dict())
        log.info("  Loaded pre-trained encoder weights.")
    else:
        log.info("  Using random encoder initialization.")

    train_labels = get_hard_labels(train_df, thresholds)
    class_weights = compute_class_weight_tensors(train_labels, device)
    criterion = SoftVADLoss(class_weights=class_weights, label_smooth=0.1)

    def make_optimizer(freeze_encoders: bool) -> torch.optim.Optimizer:
        for name in MODALITIES:
            for p in model.encoders[name].parameters():
                p.requires_grad = not freeze_encoders
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    optimizer = make_optimizer(freeze_encoders=(pretrained_encoders is not None and freeze_epochs > 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1, best_state, no_improve = -1.0, None, 0

    for epoch in range(epochs):
        # Unfreeze encoders after freeze_epochs
        if pretrained_encoders is not None and epoch == freeze_epochs:
            log.info("  Epoch %d: unfreezing encoders.", epoch + 1)
            optimizer = make_optimizer(freeze_encoders=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)

        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            summary = batch["summary"].to(device)
            bfi     = batch["bfi"].to(device)
            labels  = batch["labels"].to(device)
            seqs    = {k: v.to(device) for k, v in batch["sequences"].items()} \
                      if batch["sequences"] is not None else None
            logits = model(summary, seqs, bfi)
            soft = make_one_hot_soft(labels, device)
            dim_mask = (labels >= 0)
            sw = batch["weight"].to(device)
            loss = criterion(logits, labels, soft_targets=soft,
                             sample_weights=sw[:, 0], dim_mask=dim_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                summary = batch["summary"].to(device)
                bfi     = batch["bfi"].to(device)
                seqs    = {k: v.to(device) for k, v in batch["sequences"].items()} \
                          if batch["sequences"] is not None else None
                logits = model(summary, seqs, bfi)
                all_preds.append(logits.argmax(-1).cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        preds_arr  = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        f1s = []
        for d in range(3):
            valid = labels_arr[:, d] >= 0
            if valid.sum() == 0:
                f1s.append(0.0)
            else:
                f1s.append(float(f1_score(labels_arr[valid, d], preds_arr[valid, d],
                                          average="macro", zero_division=0)))
        val_f1 = float(np.mean(f1s))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stop at epoch %d (val F1=%.4f)", epoch + 1, best_val_f1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # Test evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            summary = batch["summary"].to(device)
            bfi     = batch["bfi"].to(device)
            seqs    = {k: v.to(device) for k, v in batch["sequences"].items()} \
                      if batch["sequences"] is not None else None
            logits = model(summary, seqs, bfi)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    test_f1s = []
    for d in range(3):
        valid = labels_arr[:, d] >= 0
        if valid.sum() == 0:
            test_f1s.append(0.0)
        else:
            test_f1s.append(float(f1_score(labels_arr[valid, d], preds_arr[valid, d],
                                            average="macro", zero_division=0)))
    return {
        "val_f1":  round(best_val_f1, 4),
        "test_f1": round(float(np.mean(test_f1s)), 4),
        "v_f1":    round(test_f1s[0], 4),
        "a_f1":    round(test_f1s[1], 4),
        "d_f1":    round(test_f1s[2], 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="data/mumt/dataset_15s.pkl")
    parser.add_argument("--pool",      default="data/mumt/augmented_pool.pkl")
    parser.add_argument("--out",       default="results/pretrain_comparison.csv")
    parser.add_argument("--encoder",   default="conv1d", choices=["conv1d", "gru"])
    parser.add_argument("--enc-dim",   type=int,   default=32)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--pretrain-epochs", type=int, default=100,
                        help="Number of contrastive pre-training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--patience",  type=int,   default=40)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="NT-Xent temperature (lower = harder negatives)")
    parser.add_argument("--freeze-epochs", type=int, default=20,
                        help="Epochs to train with frozen encoders before unfreezing")
    parser.add_argument("--seq-aug",   action="store_true")
    parser.add_argument("--test-task", default="T3")
    parser.add_argument("--compare-random", action="store_true",
                        help="Also run fine-tuning with random init for comparison")
    parser.add_argument("--save-weights", default="",
                        help="Path to save pre-trained encoder state_dict")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    df   = pd.read_pickle(args.dataset)
    pool = pd.read_pickle(args.pool)
    log.info("Dataset: %d windows  |  Pool: %d windows", len(df), len(pool))

    key_order = build_summary_key_order(df)

    train_df, val_df, test_df = task_split(df, test_task=args.test_task)
    log.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # Fit sequence scalers on training labeled data only
    seq_scalers = fit_seq_scalers(train_df)

    # Pre-train on full pool (all tasks — no label leakage since no labels used)
    pretrained_encoders = pretrain_encoders(
        pool=pool,
        seq_scalers=seq_scalers,
        encoder_type=args.encoder,
        enc_dim=args.enc_dim,
        epochs=args.pretrain_epochs,
        batch_size=args.batch,
        lr=args.lr,
        temperature=args.temperature,
        device=device,
    )

    if args.save_weights:
        state = {mod: enc.state_dict() for mod, enc in pretrained_encoders.items()}
        Path(args.save_weights).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.save_weights)
        log.info("Pre-trained weights saved → %s", args.save_weights)

    # Fine-tune with pre-trained encoders
    log.info("=== Fine-tuning with pre-trained encoders (encoder=%s, seq_aug=%s) ===",
             args.encoder, args.seq_aug)
    result_pretrained = finetune(
        train_df=train_df, val_df=val_df, test_df=test_df,
        key_order=key_order,
        encoder_type=args.encoder,
        enc_dim=args.enc_dim,
        dropout=args.dropout,
        pretrained_encoders=pretrained_encoders,
        epochs=args.finetune_epochs,
        batch_size=16,
        lr=args.lr,
        patience=args.patience,
        seq_aug=args.seq_aug,
        device=device,
        freeze_epochs=args.freeze_epochs,
    )
    log.info("  Pretrained: V=%.4f  A=%.4f  D=%.4f  Mean=%.4f  (val=%.4f)",
             result_pretrained["v_f1"], result_pretrained["a_f1"],
             result_pretrained["d_f1"], result_pretrained["test_f1"],
             result_pretrained["val_f1"])

    records = [{"encoder": args.encoder, "init": "pretrained", "seq_aug": args.seq_aug,
                **result_pretrained}]

    # Optionally compare with random init
    if args.compare_random:
        log.info("=== Fine-tuning with random init (encoder=%s, seq_aug=%s) ===",
                 args.encoder, args.seq_aug)
        result_random = finetune(
            train_df=train_df, val_df=val_df, test_df=test_df,
            key_order=key_order,
            encoder_type=args.encoder,
            enc_dim=args.enc_dim,
            dropout=args.dropout,
            pretrained_encoders=None,
            epochs=args.finetune_epochs,
            batch_size=16,
            lr=args.lr,
            patience=args.patience,
            seq_aug=args.seq_aug,
            device=device,
            freeze_epochs=0,
        )
        log.info("  Random init: V=%.4f  A=%.4f  D=%.4f  Mean=%.4f  (val=%.4f)",
                 result_random["v_f1"], result_random["a_f1"],
                 result_random["d_f1"], result_random["test_f1"],
                 result_random["val_f1"])
        records.append({"encoder": args.encoder, "init": "random", "seq_aug": args.seq_aug,
                        **result_random})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)

    print("\n=== Pre-training comparison ===")
    print(pd.DataFrame(records)[["encoder", "init", "seq_aug",
                                  "v_f1", "a_f1", "d_f1", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
