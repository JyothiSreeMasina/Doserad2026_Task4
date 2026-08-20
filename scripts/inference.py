"""Single-control-point inference: CT + beam params -> dose prediction (Gy).

This is the unit the challenge times against the 1 second/beam limit -- for
photon/VMAT, "beam" in the scoring sense is one control point (see
DoseRAD2026_Complete_Reference.md's "Photon: per control point" note, and
`src/data/hf_dataset.py`'s docstring: 40,500 training samples = 75 patients x
3 beams x 180 control points, i.e. each control point is independently
evaluated -- not summed into a full arc first).

Usage:
    python scripts/inference.py --config configs/task1_photon_ct.yaml \
        --checkpoint checkpoints/task1_photon_ct/last.pt \
        --patient 1ABB006 --beam 0 --cp 0 \
        [--benchmark N]   # time N repeated forward passes after warmup
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.beam_encoder import PhotonBeamEncoder
from src.data.hf_dataset import DoseRAD2026Dataset
from src.data.transforms import BodyMask, CenterCropOrPad, Compose, NormalizeCT, ResampleToSpacing
from src.models.model_factory import build_model


def build_inference_dataset(dcfg):
    # Deliberately NO ScaleDose here: this path only ever reads `dose` to
    # report ground truth for comparison, never as a training target, so
    # dose stays in real Gy. Model output is unscaled explicitly below.
    transform = Compose([
        ResampleToSpacing(target_spacing=tuple(dcfg["target_spacing"])),
        CenterCropOrPad(target_shape=tuple(dcfg["target_shape"])),
        BodyMask(),
        NormalizeCT(),
    ])
    return DoseRAD2026Dataset(
        data_root=dcfg["data_root"], task=dcfg["task"], split=dcfg["split"],
        transform=transform, cache_size=dcfg["cache_size"],
    )


@torch.no_grad()
def predict_dose(model, encoder, sample, device, dose_scale, amp=True):
    """One control point -> predicted dose volume in real Gy, (D, H, W)."""
    ct = sample["ct"].unsqueeze(0).to(device)
    beam_params = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in sample["beam_params"].items()}
    beam_mask = encoder.encode(beam_params).unsqueeze(0)

    with torch.autocast(device_type="cuda", enabled=(amp and device == "cuda")):
        pred = model(ct, beam_mask)
    return (pred.squeeze(0).squeeze(0) / dose_scale).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1_photon_ct.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--beam", type=int, default=0)
    parser.add_argument("--cp", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", type=int, default=0, help="If >0, time this many repeated forward passes after warmup")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    dataset = build_inference_dataset(dcfg)
    idx = dataset.index.index((args.patient, args.beam, args.cp))
    sample = dataset[idx]

    model = build_model(cfg["model"]).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {state['epoch']}")

    volume_shape = tuple(sample["ct"].shape[-3:])
    encoder = PhotonBeamEncoder(volume_shape, sample["ct_spacing"], sample["ct_origin"])
    encoder._coords_xyz = encoder._coords_xyz.to(args.device)

    # Warmup: first call(s) pay one-time CUDA/cudnn kernel-selection cost
    # that a real deployed instance won't repeatedly pay across beams.
    for _ in range(3):
        predict_dose(model, encoder, sample, args.device, dcfg["dose_scale"])
    if args.device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    dose_pred = predict_dose(model, encoder, sample, args.device, dcfg["dose_scale"])
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Single control point: {elapsed:.4f}s (limit: 1.0s/beam)")
    print(f"Predicted dose: max={dose_pred.max().item():.3e} Gy, sum={dose_pred.sum().item():.3e} Gy")

    if args.benchmark > 0:
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.benchmark):
            predict_dose(model, encoder, sample, args.device, dcfg["dose_scale"])
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        per_call = elapsed / args.benchmark
        print(f"Benchmark: {args.benchmark} calls in {elapsed:.2f}s -> {per_call:.4f}s/call "
              f"({'PASS' if per_call <= 1.0 else 'FAIL'}, limit 1.0s)")


if __name__ == "__main__":
    main()
