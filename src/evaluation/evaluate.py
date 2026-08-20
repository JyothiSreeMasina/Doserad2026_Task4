"""Runs the official DoseRAD2026 metrics on held-out validation patients.

Usage:
    python src/evaluation/evaluate.py --checkpoint checkpoints/task1_photon_ct/best.pt
    python src/evaluation/evaluate.py --checkpoint ... --cp_stride 20 --max_patients 2   # quick spot-check

Level 2 metrics (stratified_plan_mae, gamma_index_pass_rate) use each
patient's own max full-plan dose as a prescription stand-in, and
dvh_clinical_score is reported as None -- see src/evaluation/metrics.py's
module docstring for why (no prescription/RTSTRUCT data in the local
training set).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.beam_encoder import PhotonBeamEncoder
from src.data.hf_dataset import DoseRAD2026Dataset, split_patient_ids
from src.data.transforms import BodyMask, BodyMaskMR, CenterCropOrPad, Compose, NormalizeCT, NormalizeMR, ResampleToSpacing, ScaleDose
from src.evaluation.metrics import gamma_index_3d, idd_curve_distance, masked_mae, stratified_plan_mae
from src.models.model_factory import build_model


def build_val_dataset(dcfg):
    modality = dcfg.get("modality", "ct")
    mask_transform, normalize_transform = (BodyMaskMR(), NormalizeMR()) if modality == "mr" else (BodyMask(), NormalizeCT())
    transform = Compose([
        ResampleToSpacing(target_spacing=tuple(dcfg["target_spacing"])),
        CenterCropOrPad(target_shape=tuple(dcfg["target_shape"])),
        mask_transform,
        normalize_transform,
        ScaleDose(scale=dcfg["dose_scale"]),
    ])
    dataset = DoseRAD2026Dataset(
        data_root=dcfg["data_root"], task=dcfg["task"], split=dcfg["split"], modality=modality,
        transform=transform, cache_size=dcfg["cache_size"],
    )
    _, val_ids = split_patient_ids(dataset.index, dcfg["val_fraction"], dcfg["seed"])
    return dataset, val_ids


def _mean_ignore_nan(values):
    clean = [v for v in values if v == v]  # v == v is False for NaN
    return sum(clean) / len(clean) if clean else float("nan")


@torch.no_grad()
def evaluate(model, dataset, val_ids, device, high_dose_frac=0.1,
             gamma_dose_percent=1.0, gamma_distance_mm=1.0, cp_stride=1):
    """Runs Level 1 metrics per control point and Level 2 metrics per
    full-plan (sum of a patient's control-point doses across all beams).
    """
    model.eval()
    encoder_cache = {}
    level1 = defaultdict(list)
    plans = {}  # patient_id -> {"pred", "target", "spacing"}

    by_patient = defaultdict(list)
    for i, (pid, beam_idx, cp_idx) in enumerate(dataset.index):
        if pid in val_ids:
            by_patient[pid].append(i)

    for pid, idxs in by_patient.items():
        idxs = idxs[::cp_stride]
        print(f"Evaluating {pid} ({len(idxs)} control points)...")
        plan_pred, plan_target, spacing = None, None, None

        for i in idxs:
            sample = dataset[i]
            ct = sample["ct"].unsqueeze(0).to(device)
            dose = sample["dose"].to(device)
            spacing = sample["ct_spacing"]

            if pid not in encoder_cache:
                volume_shape = tuple(sample["ct"].shape[-3:])
                enc = PhotonBeamEncoder(volume_shape, sample["ct_spacing"], sample["ct_origin"])
                enc.to(device)
                encoder_cache[pid] = enc
            encoder = encoder_cache[pid]

            beam_params = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in sample["beam_params"].items()}
            beam_mask = encoder.encode(beam_params).unsqueeze(0)
            pred = model(ct, beam_mask).squeeze(0)  # (1, D, H, W)

            level1["masked_mae"].append(masked_mae(pred, dose, high_dose_frac=high_dose_frac))
            level1["idd_curve_distance"].append(idd_curve_distance(pred, dose, encoder, beam_params))

            if plan_pred is None:
                plan_pred, plan_target = pred.clone(), dose.clone()
            else:
                plan_pred += pred
                plan_target += dose

        plans[pid] = {"pred": plan_pred, "target": plan_target, "spacing": spacing}

    level2 = defaultdict(list)
    for plan in plans.values():
        prescription = plan["target"].max().item()  # approximation -- see metrics.py module docstring
        strat = stratified_plan_mae(plan["pred"], plan["target"], prescription=prescription)
        level2["stratified_plan_mae"].append(strat["mean"])
        gamma = gamma_index_3d(
            plan["pred"], plan["target"], plan["spacing"], prescription=prescription,
            dose_percent=gamma_dose_percent, distance_mm=gamma_distance_mm,
        )
        level2["gamma_index_pass_rate"].append(gamma)

    summary = {name: _mean_ignore_nan(values) for name, values in {**level1, **level2}.items()}
    summary["dvh_clinical_score"] = None  # requires PTV/OAR structures not present locally -- see metrics.py
    summary["n_val_patients"] = len(plans)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1_photon_ct.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cp_stride", type=int, default=1, help="Evaluate every Nth control point (for quick spot-checks)")
    parser.add_argument("--max_patients", type=int, default=None, help="Cap number of val patients evaluated")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, val_ids = build_val_dataset(cfg["data"])
    if args.max_patients:
        val_ids = val_ids[: args.max_patients]
    print(f"Validation patients ({len(val_ids)}): {val_ids}")

    model = build_model(cfg["model"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"Loaded checkpoint from epoch {state['epoch']}")

    summary = evaluate(
        model, dataset, val_ids, device,
        high_dose_frac=cfg["loss"]["high_dose_frac"], cp_stride=args.cp_stride,
    )

    print("\n=== Validation metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
