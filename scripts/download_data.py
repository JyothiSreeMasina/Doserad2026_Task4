"""Download patients from the DoseRAD2026 HF dataset to local disk.

Usage:
    python scripts/download_data.py --patients 1ABB006 1ABB011 --task photon --split training
    python scripts/download_data.py --task photon --split training   # all patients in the split
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "LMUK-RADONC-PHYS-RES/DoseRAD2026"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", nargs="+", default=None, help="Patient IDs, e.g. 1ABB006 1ABB011 (omit for all)")
    parser.add_argument("--task", default="photon", choices=["photon", "proton"])
    parser.add_argument("--split", default="training", choices=["training", "testing"])
    parser.add_argument("--out", default=None, help="Output dir (default: data/<task>_ct/)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "data" / f"{args.task}_ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.patients:
        patterns = [f"{args.task}/{args.split}/{pid}/**" for pid in args.patients]
        label = f"{len(args.patients)} patient(s)"
    else:
        patterns = [f"{args.task}/{args.split}/**"]
        label = f"all patients in {args.task}/{args.split}"

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=out_dir,
    )
    print(f"Downloaded {label} to {out_dir}")


if __name__ == "__main__":
    main()
