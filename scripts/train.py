"""Train the Task 1 (photon/CT) dose model.

Usage:
    python scripts/train.py --config configs/task1_photon_ct.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.beam_encoder import PhotonBeamEncoder, ProtonBeamEncoder
from src.data.hf_dataset import DoseRAD2026Dataset, indices_for_patients, split_patient_ids
from src.data.transforms import (
    BodyMask, BodyMaskMR, CenterCropOrPad, Compose, NormalizeCT, NormalizeMR,
    RandomIntensityShift, ResampleToSpacing, ScaleDose,
)
from src.models.model_factory import build_model
from src.training.losses import DoseLoss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1_photon_ct.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dcfg = cfg["data"]
    modality = dcfg.get("modality", "ct")
    mask_transform, normalize_transform = (BodyMaskMR(), NormalizeMR()) if modality == "mr" else (BodyMask(), NormalizeCT())
    base_transforms = [
        ResampleToSpacing(target_spacing=tuple(dcfg["target_spacing"])),
        CenterCropOrPad(target_shape=tuple(dcfg["target_shape"])),
        mask_transform,
        normalize_transform,
        ScaleDose(scale=dcfg["dose_scale"]),
    ]
    # Separate Dataset instances for train/val (not one Dataset shared via
    # Subset) so RandomIntensityShift -- an augmentation -- only ever runs on
    # the train split. Both instances scan the same directory in the same
    # sorted order, so their `.index` lists line up and split_patient_ids'
    # indices apply to either one.
    train_transform = Compose(base_transforms + [RandomIntensityShift()])
    val_transform = Compose(base_transforms)

    train_dataset = DoseRAD2026Dataset(
        data_root=dcfg["data_root"], task=dcfg["task"], split=dcfg["split"], modality=modality,
        transform=train_transform, cache_size=dcfg["cache_size"],
    )
    val_dataset = DoseRAD2026Dataset(
        data_root=dcfg["data_root"], task=dcfg["task"], split=dcfg["split"], modality=modality,
        transform=val_transform, cache_size=dcfg["cache_size"],
    )
    print(f"Dataset: {len(train_dataset)} control points across {len({t[0] for t in train_dataset.index})} patients")

    train_ids, val_ids = split_patient_ids(train_dataset.index, dcfg["val_fraction"], dcfg["seed"])
    train_set = Subset(train_dataset, indices_for_patients(train_dataset.index, train_ids))
    val_set = Subset(val_dataset, indices_for_patients(val_dataset.index, val_ids))
    print(f"Split: {len(train_ids)} train patients / {len(val_ids)} val patients")
    print(f"  val patients: {val_ids}")

    identity_collate = lambda batch: batch[0]  # noqa: E731 -- batch size is fixed at 1, see Trainer docstring
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=dcfg["num_workers"], collate_fn=identity_collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=dcfg["num_workers"], collate_fn=identity_collate,
    )

    model = build_model(cfg["model"])
    loss_fn = DoseLoss(**cfg["loss"])

    ocfg = cfg["optim"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"], weight_decay=ocfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ocfg["epochs"])

    beam_encoder_cls = ProtonBeamEncoder if dcfg["task"] == "proton" else PhotonBeamEncoder
    trainer = Trainer(
        model, train_loader, val_loader, loss_fn, optimizer,
        device=device,
        checkpoint_dir=cfg["checkpoint_dir"],
        scheduler=scheduler,
        grad_clip=ocfg["grad_clip"],
        log_every=cfg["log_every"],
        amp=ocfg["amp"],
        accum_steps=ocfg["accum_steps"],
        save_every_steps=cfg.get("save_every_steps", 2000),
        beam_encoder_cls=beam_encoder_cls,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    trainer.fit(ocfg["epochs"], start_epoch=start_epoch)


if __name__ == "__main__":
    main()
