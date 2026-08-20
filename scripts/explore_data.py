"""One-off exploration of a single patient's CT, dose, and beam JSON.

Usage:
    python scripts/explore_data.py --patient 1ABB006
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def describe_image(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    print(f"  file: {path.name}")
    print(f"  size (x,y,z): {img.GetSize()}")
    print(f"  spacing (mm): {img.GetSpacing()}")
    print(f"  origin: {img.GetOrigin()}")
    print(f"  direction: {img.GetDirection()}")
    print(f"  dtype: {arr.dtype}, array shape (z,y,x): {arr.shape}")
    print(f"  value range: [{arr.min():.4f}, {arr.max():.4f}], mean: {arr.mean():.4f}")
    return img, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="1ABB006")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else Path(__file__).resolve().parents[2] / "data" / "photon_ct"
    pdir = root / "photon" / "training" / args.patient

    print("=" * 70)
    print(f"Patient: {args.patient}  (dir: {pdir})")
    print("=" * 70)

    print("\n--- CT image ---")
    ct_img, ct_arr = describe_image(pdir / "image" / "ct.mha")

    print("\n--- MR image ---")
    mr_img, mr_arr = describe_image(pdir / "image" / "mr.mha")

    dose_dir = pdir / "dose"
    dose_files = sorted(dose_dir.glob("*.mha"))
    print(f"\n--- Dose files ---")
    print(f"  total dose files: {len(dose_files)}")
    print(f"  first few: {[f.name for f in dose_files[:5]]}")

    print(f"\n--- Dose sample: {dose_files[0].name} ---")
    dose_img, dose_arr = describe_image(dose_files[0])
    print(f"  same spacing as CT? {dose_img.GetSpacing() == ct_img.GetSpacing()}")
    print(f"  same size as CT? {dose_img.GetSize() == ct_img.GetSize()}")

    print(f"\n--- Dose sample 2: {dose_files[-1].name} ---")
    describe_image(dose_files[-1])

    json_path = pdir / f"{args.patient}.json"
    print(f"\n--- Beam config JSON: {json_path.name} ---")
    with open(json_path) as f:
        beam_cfg = json.load(f)
    print(f"  top-level type: {type(beam_cfg)}")
    if isinstance(beam_cfg, dict):
        print(f"  top-level keys: {list(beam_cfg.keys())}")
        print(json.dumps(beam_cfg, indent=2)[:3000])
    elif isinstance(beam_cfg, list):
        print(f"  list length: {len(beam_cfg)}")
        print(json.dumps(beam_cfg[0], indent=2)[:3000])


if __name__ == "__main__":
    main()
