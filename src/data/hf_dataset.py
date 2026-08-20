"""Local-disk dataset for DoseRAD2026.

One sample = one (patient, beam, control_point) triple: a patient's CT volume
paired with a single control point's beam geometry and that control point's
Monte Carlo dose grid as the target.

Granularity confirmed from DoseRAD2026_Complete_Reference.md: the training
set has 40,500 photon beam segments across 75 patients (40500 / 75 = 540 =
3 beams x 180 control points per patient), so each control point is its own
independent unit -- dose grids are not summed across control points.
"""
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class DoseRAD2026Dataset(Dataset):
    def __init__(self, data_root, task="photon", split="training", modality="ct", transform=None, cache_size=4):
        self.root = Path(data_root) / task / split
        self.task = task  # "photon" or "proton" -- selects control-point vs beam/ray/beamlet indexing below
        self.modality = modality  # "ct" or "mr" -- same beam JSON/dose targets, different image/<modality>.mha
        self.transform = transform
        self._load_ct = lru_cache(maxsize=cache_size)(self._load_ct_uncached)

        self.index = []  # photon: (pid, beam_idx, cp_idx); proton: (pid, beam_idx, ray_idx, beamlet_idx)
        self._cp_lookup = {}  # patient_id -> {index_key: (beam_dict, cp_or_beamlet_dict)}

        for pdir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            pid = pdir.name
            json_path = pdir / f"{pid}.json"
            if not json_path.exists():
                continue

            with open(json_path) as f:
                beams = json.load(f)["beams"]

            lookup = {}
            if self.task == "proton":
                # Proton JSON nests one more level than photon: beam -> rays
                # (individual scanned spots, each with its own ray_source/
                # ray_target -- see beam_encoder.py's ProtonBeamEncoder
                # docstring) -> beamlets (one per energy layer). Dose files
                # are named by that full (beam, ray, beamlet) triple, unlike
                # photon's (beam, control_point) pair.
                for beam in beams:
                    beam_idx = beam["beam_idx"]
                    for ray in beam["rays"]:
                        ray_idx = ray["ray_idx"]
                        for bl in ray["beamlets"]:
                            bl_idx = bl["beamlet_idx"]
                            dose_path = pdir / "dose" / f"Dose_B{beam_idx}_R{ray_idx}_L{bl_idx}.mha"
                            if dose_path.exists():
                                lookup[(beam_idx, ray_idx, bl_idx)] = (beam, ray, bl)
                                self.index.append((pid, beam_idx, ray_idx, bl_idx))
            else:
                for beam in beams:
                    beam_idx = beam["beam_idx"]
                    for cp in beam["control_points"]:
                        cp_idx = cp["cp_idx"]
                        dose_path = pdir / "dose" / f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha"
                        if dose_path.exists():
                            lookup[(beam_idx, cp_idx)] = (beam, cp)
                            self.index.append((pid, beam_idx, cp_idx))
            self._cp_lookup[pid] = lookup

    def __len__(self):
        return len(self.index)

    def _load_ct_uncached(self, patient_id):
        # Kept as "ct" throughout the sample dict / downstream code (trainer,
        # evaluate, docker/process.py) even for modality="mr" -- every
        # consumer treats it as "the input image channel" generically, and
        # renaming would be pure churn for zero behavior change.
        ct_path = self.root / patient_id / "image" / f"{self.modality}.mha"
        img = sitk.ReadImage(str(ct_path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # sitk spacing is (x,y,z); array axes are (z,y,x) -- reverse to match.
        spacing_zyx = tuple(reversed(img.GetSpacing()))
        return arr, spacing_zyx, img.GetOrigin()

    def __getitem__(self, idx):
        if self.task == "proton":
            pid, beam_idx, ray_idx, bl_idx = self.index[idx]
            beam, ray, bl = self._cp_lookup[pid][(beam_idx, ray_idx, bl_idx)]
            dose_path = self.root / pid / "dose" / f"Dose_B{beam_idx}_R{ray_idx}_L{bl_idx}.mha"
            beam_params = {
                "ray_source": torch.tensor(ray["ray_source"], dtype=torch.float32),
                "ray_target": torch.tensor(ray["ray_target"], dtype=torch.float32),
                "energy": float(bl["energy"]),
            }
        else:
            pid, beam_idx, cp_idx = self.index[idx]
            beam, cp = self._cp_lookup[pid][(beam_idx, cp_idx)]
            dose_path = self.root / pid / "dose" / f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha"
            beam_params = {
                "gantry_angle": float(cp["gantry_angle"]),
                "iso_center": torch.tensor(beam["iso_center"], dtype=torch.float32),
                "SAD": float(beam["SAD"]),
                "mlc_left_int_mm": torch.tensor(cp["mlc_left_int_mm"], dtype=torch.float32),
                "mlc_right_int_mm": torch.tensor(cp["mlc_right_int_mm"], dtype=torch.float32),
            }

        ct, ct_spacing, ct_origin = self._load_ct(pid)
        dose = sitk.GetArrayFromImage(sitk.ReadImage(str(dose_path))).astype(np.float32)

        sample = {
            "patient_id": pid,
            "ct": torch.from_numpy(ct).unsqueeze(0),
            "dose": torch.from_numpy(dose).unsqueeze(0),
            "ct_spacing": ct_spacing,
            "ct_origin": ct_origin,
            "beam_params": beam_params,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def split_patient_ids(index, val_fraction, seed):
    """Deterministic patient-level train/val split, shared by scripts/train.py
    and src/evaluation/evaluate.py so both see exactly the same held-out
    patients for a given seed.

    Held-out *patients*, not held-out control points -- a random per-CP split
    would leak the same patient's anatomy into both train and val (540
    control points share one CT), making val metrics a poor generalization
    signal. Works for both photon's (pid, beam_idx, cp_idx) and proton's
    (pid, beam_idx, ray_idx, beamlet_idx) index tuples -- only element 0 is used.
    """
    patient_ids = sorted({t[0] for t in index})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n_val = max(1, round(len(patient_ids) * val_fraction))
    val_ids = set(patient_ids[:n_val])
    train_ids = set(patient_ids[n_val:])
    return sorted(train_ids), sorted(val_ids)


def indices_for_patients(index, patient_ids):
    """Indices into `index` (or a Dataset built from the same directory scan)
    belonging to the given patient IDs. Works for both photon's 3-tuple and
    proton's 4-tuple index entries -- only element 0 is used."""
    wanted = set(patient_ids)
    return [i for i, t in enumerate(index) if t[0] in wanted]
