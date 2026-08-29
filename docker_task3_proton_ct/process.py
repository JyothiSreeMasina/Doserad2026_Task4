"""DoseRAD2026 Task 3 (proton/CT) inference for Grand Challenge's `invoke` API.

Reads the batched input contract at /input (up to 10 source CT images plus
one beam-level metadata JSON covering all of them), runs one model forward
pass per beamlet (the unit the challenge times against the 1s/beam limit --
see DoseRAD2026_Complete_Reference.md's "Proton: per pencil beamlet" note),
and writes the ten stacked-dose-map output slots at /output.

Mirrors docker/process.py (Task 1, photon/CT) closely -- same wrapper
architecture (N repeated CT image slots + 1 stacked beam-level metadata JSON
in, N stacked dose map slots out), confirmed identical for Task 3 via the
real Grand Challenge "Preliminary test - Proton dose on CT" algorithm page
(Inputs: "Radiation-Dose Calculation Source CT Image" x10 + "Stacked Proton
Beam-Level Metadata"; Outputs: "Stacked Radiation-Dose Map" x10 -- same
socket names/counts as Task 1, just "photon"->"proton" in the metadata
socket). Only the beam content differs: proton is pencil-beam-scanning
(beam -> ray -> beamlet/energy-layer, ProtonBeamEncoder), not VMAT control
points (beam -> control_point, PhotonBeamEncoder).

Metadata schema, updated after a real submission (2026-08-25, result
78dc528f): image_entry["beams"]/beam["rays"]/ray["beamlets"] and each
beamlet's output_info.{output_file_idx, idx_in_output} are CONFIRMED --
that first real /invoke ran the entire slot-size precompute pass (every
beamlet in the file) without a KeyError. The only failure was an optional
`beam_idx`/`ray_idx` field used purely for a log line, not present in the
real file -- fixed by falling back to our own loop index instead of
depending on those keys (see run()'s main loop below). STILL UNVERIFIED
(the crash happened before the model ever ran a real prediction): ray's
ray_source/ray_target and beamlet's energy/minimum_cutoff -- still
inferred from the training-data JSON schema (src/data/hf_dataset.py) by
analogy with Task 1's confirmed wrapper. If the next submission fails
differently (e.g. a KeyError inside predict_beamlet or on minimum_cutoff),
check those field names next.

Preprocessing (resample/crop-pad/body-mask/normalize) reuses the exact same
`src/data/transforms.py` classes used during training -- critical, since any
drift between train-time and inference-time preprocessing would silently
invalidate the model's predictions. The one extra step needed only at
inference time: our internal grid (176x512x512 @ 3x1x1mm, center-cropped/
padded) does not match the original input image's native grid, but the
challenge requires "Output dose grid must match input image exactly
(identical size, spacing, origin, direction)". That's handled by building a
proper SimpleITK image on our internal grid (using the origin/spacing our
transform pipeline already tracks) and calling sitk.Resample with the
*original* CT image as the reference grid -- this correctly undoes both the
crop/pad and any resample in one physically-grounded operation, rather than
hand-rolling inverse array slicing. Voxels outside our modeled window (if a
patient's native FOV is larger) default to zero dose.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from src.data.beam_encoder import ProtonBeamEncoder
from src.data.transforms import BodyMask, CenterCropOrPad, Compose, NormalizeCT, ResampleToSpacing
from src.models.model_factory import build_model

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
CONFIG_PATH = Path("/opt/app/configs/task3_proton_ct.yaml")
MODEL_DIR = Path("/opt/ml/model")

NUM_OUTPUT_FILES = 10
# Same socket name as Task 1 -- confirmed via the real GC Task 3 algorithm
# page, not proton-specific naming.
CT_DIR_BASE = "radiation-dose-calculation-source-ct-image"
METADATA_JSON_NAME = "stacked-proton-beam-level-metadata"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def init_model(device):
    """Loads the trained DoseRADModel checkpoint. Called once at server
    startup (see app.py) -- must not run per-invoke, or the per-beam timeout
    budget would be spent loading weights instead of predicting dose."""
    cfg = load_config()
    model = build_model(cfg["model"]).to(device)

    ckpt_candidates = sorted(MODEL_DIR.rglob("*.pt"))
    if not ckpt_candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under {MODEL_DIR}")
    ckpt_path = ckpt_candidates[0]
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"Loaded checkpoint {ckpt_path} (epoch {state.get('epoch', '?')})")

    _warmup(model, cfg, device)
    return model, cfg


def _warmup(model, cfg, device):
    """Pays CUDA/cuDNN kernel-selection cost once at startup instead of on
    the first real /invoke call -- see docker/process.py's _warmup()
    docstring (Task 1) for the full rationale; identical concern here, just
    exercising ProtonBeamEncoder.encode() instead of PhotonBeamEncoder's.
    """
    dcfg = cfg["data"]
    target_shape = tuple(dcfg["target_shape"])
    target_spacing = tuple(dcfg["target_spacing"])
    amp = cfg["optim"].get("amp", True)

    dummy_ct = torch.zeros((1, 1, *target_shape), device=device)
    dummy_body_mask = torch.ones((1, *target_shape), device=device)  # matches encode()'s call shape; see body_mask docstring
    encoder = ProtonBeamEncoder(target_shape, target_spacing, origin=(0.0, 0.0, 0.0))
    encoder.to(device)
    # Roughly centered dummy ray along z with a mid-range energy -- exact
    # values don't matter for warmup, just needs to exercise the same
    # tensor-op shapes as a real encode() call.
    dummy_beam_params = {
        "ray_source": torch.tensor([0.0, -1000.0, 0.0], device=device),
        "ray_target": torch.tensor([0.0, 0.0, 0.0], device=device),
        "energy": 100.0,
    }

    with torch.no_grad():
        for _ in range(3):
            beam_mask = encoder.encode(dummy_beam_params, body_mask=dummy_body_mask, ct=dummy_ct[0]).unsqueeze(0)
            with torch.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
                model(dummy_ct, beam_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("Warmup complete")


def load_sitk_image(location):
    mha_files = glob.glob(str(location / "*.mha"))
    if not mha_files:
        raise FileNotFoundError(f"No .mha file found in {location}")
    return sitk.ReadImage(mha_files[0])


def preprocess_ct(ct_image, target_spacing, target_shape):
    """Mirrors DoseRAD2026Dataset.__getitem__ + the training transform
    pipeline exactly (minus ScaleDose, which only applies to dose targets),
    so the model sees input distributed identically to training.
    """
    arr = sitk.GetArrayFromImage(ct_image).astype(np.float32)  # (z, y, x)
    spacing_zyx = tuple(reversed(ct_image.GetSpacing()))
    sample = {
        "ct": torch.from_numpy(arr).unsqueeze(0),
        "ct_spacing": spacing_zyx,
        "ct_origin": ct_image.GetOrigin(),
    }
    transform = Compose([
        ResampleToSpacing(target_spacing=target_spacing),
        CenterCropOrPad(target_shape=target_shape),
        BodyMask(),
        NormalizeCT(),
    ])
    return transform(sample)


def restore_dose_to_original_grid(dose_tensor, minimum_cutoff, restore_cropper, sample, original_image):
    """dose_tensor: (D, H, W) float32 tensor on our internal (cropped/
    padded, possibly resampled) grid. Returns a (D, H, W) numpy array on
    `original_image`'s exact grid (size, spacing, origin, direction), with
    `minimum_cutoff` applied. Returns a bare array rather than an sitk.Image
    -- run() writes one beamlet at a time into a preallocated buffer for the
    whole output slot; see docker/process.py's identical helper (Task 1) for
    the full rationale (fast-path GPU-only restore when spacing is
    unchanged, falls back to sitk.Resample only if it genuinely differs).
    """
    dose_tensor = torch.where(dose_tensor < minimum_cutoff, torch.zeros_like(dose_tensor), dose_tensor)
    original_shape_zyx = tuple(reversed(original_image.GetSize()))
    spacing_matches = tuple(round(s, 6) for s in sample["ct_spacing"]) == \
        tuple(round(s, 6) for s in reversed(original_image.GetSpacing()))

    if spacing_matches:
        restored = restore_cropper.restore(dose_tensor.unsqueeze(0), original_shape_zyx).squeeze(0)
        return restored.cpu().numpy()

    dose_img = sitk.GetImageFromArray(dose_tensor.cpu().numpy())
    dose_img.SetSpacing(tuple(reversed(sample["ct_spacing"])))
    dose_img.SetOrigin(sample["ct_origin"])
    dose_img.SetDirection(original_image.GetDirection())
    resampled = sitk.Resample(
        dose_img,
        referenceImage=original_image,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        outputPixelType=sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(resampled)


@torch.no_grad()
def predict_beamlet(model, encoder, ct_tensor, body_mask, ray, beamlet, device, dose_scale, amp):
    beam_params = {
        "ray_source": torch.tensor(ray["ray_source"], dtype=torch.float32, device=device),
        "ray_target": torch.tensor(ray["ray_target"], dtype=torch.float32, device=device),
        "energy": float(beamlet["energy"]),
    }
    beam_mask = encoder.encode(beam_params, body_mask=body_mask, ct=ct_tensor[0]).unsqueeze(0)
    with torch.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
        pred = model(ct_tensor, beam_mask)
    return (pred.squeeze(0).squeeze(0) / dose_scale).float()  # (D, H, W), stays on device


def _direction_3d_to_4d(direction3):
    """Extends a 3x3 row-major direction tuple to the 4x4 sitk expects for a
    4D image -- the extra axis (beamlet index) is always identity."""
    return (
        direction3[0], direction3[1], direction3[2], 0.0,
        direction3[3], direction3[4], direction3[5], 0.0,
        direction3[6], direction3[7], direction3[8], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def run(model, cfg, device):
    dcfg = cfg["data"]
    target_shape = tuple(dcfg["target_shape"])
    target_spacing = tuple(dcfg["target_spacing"])
    dose_scale = dcfg["dose_scale"]
    amp = cfg["optim"].get("amp", True)

    metadata_path = INPUT_PATH / f"{METADATA_JSON_NAME}.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} image(s)")

    restore_cropper = CenterCropOrPad(target_shape=target_shape)

    # Same output_file_idx/idx_in_output slot-binning contract as Task 1
    # (docker/process.py) -- see that file's run() docstring comment for the
    # full rationale (output_file_idx is independent of which image a
    # beamlet came from; idx_in_output's max value IS the slot's final
    # stack size, so a slot is safe to flush the instant its last index is
    # filled). UNVERIFIED here: whether output_info actually attaches at
    # the beamlet level (this code's assumption, mirroring where photon
    # attaches it at the control-point/timed-unit level) -- see module
    # docstring.
    slot_sizes = {}
    for image_entry in metadata:
        for beam in image_entry["beams"]:
            for ray in beam["rays"]:
                for bl in ray["beamlets"]:
                    oi = bl["output_info"]
                    slot_sizes[oi["output_file_idx"]] = max(
                        slot_sizes.get(oi["output_file_idx"], 0), oi["idx_in_output"] + 1
                    )

    buffers = {}        # output_file_idx -> np.ndarray, allocated on first use
    filled = {}         # output_file_idx -> count of slices written so far
    buffer_image = {}   # output_file_idx -> the original_image its data came from
    written_slots = set()

    def flush_slot(output_file_idx):
        buffer = buffers.pop(output_file_idx)
        filled.pop(output_file_idx)
        original_image = buffer_image.pop(output_file_idx)
        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_file_idx + 1}"
        os.makedirs(output_dir, exist_ok=True)
        # isVector=False required -- see docker/process.py's flush_slot()
        # docstring (Task 1) for why: GetImageFromArray's isVector
        # auto-detection would otherwise silently collapse the 4D stack to
        # 3D, dropping the beamlet axis entirely.
        stacked = sitk.GetImageFromArray(buffer, isVector=False)
        stacked.SetSpacing(tuple(original_image.GetSpacing()) + (1.0,))
        stacked.SetOrigin(tuple(original_image.GetOrigin()) + (0.0,))
        stacked.SetDirection(_direction_3d_to_4d(original_image.GetDirection()))
        sitk.WriteImage(stacked, output_dir / "output.mha", useCompression=False)
        print(f"Wrote output slot {output_file_idx + 1}: {buffer.shape[0]} slice(s)")
        written_slots.add(output_file_idx)

    for image_entry in metadata:
        image_idx = image_entry["image_file_idx"]
        location = INPUT_PATH / f"images/{CT_DIR_BASE}-{image_idx + 1}"
        original_image = load_sitk_image(location)
        print(f"Image {image_idx + 1}: size={original_image.GetSize()} spacing={original_image.GetSpacing()}")

        sample = preprocess_ct(original_image, target_spacing, target_shape)
        ct_tensor = sample["ct"].unsqueeze(0).to(device)
        body_mask_tensor = sample["body_mask"].to(device)

        volume_shape = tuple(sample["ct"].shape[-3:])
        encoder = ProtonBeamEncoder(volume_shape, sample["ct_spacing"], sample["ct_origin"])
        encoder.to(device)

        shape_zyx = tuple(reversed(original_image.GetSize()))
        n_beams = len(image_entry["beams"])
        for beam_i, beam in enumerate(image_entry["beams"]):
            for ray_i, ray in enumerate(beam["rays"]):
                n_bl = len(ray["beamlets"])
                # beam_idx/ray_idx aren't confirmed key names in the real
                # metadata (see module docstring) and aren't needed for the
                # actual computation below -- fall back to our own loop
                # index rather than depending on a JSON field for logging.
                print(f"  Beam {beam.get('beam_idx', beam_i)} ray {ray.get('ray_idx', ray_i)}: {n_bl} beamlet(s)")
                for bl in ray["beamlets"]:
                    output_info = bl["output_info"]
                    idx = output_info["output_file_idx"]
                    minimum_cutoff = float(output_info["minimum_cutoff"])
                    dose_tensor = predict_beamlet(
                        model, encoder, ct_tensor, body_mask_tensor, ray, bl, device, dose_scale, amp
                    )
                    dose_array = restore_dose_to_original_grid(
                        dose_tensor, minimum_cutoff, restore_cropper, sample, original_image
                    )
                    if idx not in buffers:
                        buffers[idx] = np.zeros((slot_sizes[idx], *shape_zyx), dtype=np.float32)
                        filled[idx] = 0
                        buffer_image[idx] = original_image
                    buffers[idx][output_info["idx_in_output"]] = dose_array
                    filled[idx] += 1
                    if filled[idx] == slot_sizes[idx]:
                        flush_slot(idx)
        print(f"Image {image_idx + 1}: {n_beams} beam(s) done")

    # Defensive fallback -- shouldn't trigger given the "no gaps" contract,
    # but if a slot's beamlets ever arrive out of the order that guarantee
    # implies, this still gets it written instead of silently dropped.
    for idx in list(buffers.keys()):
        flush_slot(idx)

    for output_index in range(NUM_OUTPUT_FILES):
        if output_index in written_slots:
            continue
        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_index + 1}"
        os.makedirs(output_dir, exist_ok=True)
        # Genuine 4D (1,1,1,1) placeholder -- matches the instructions'
        # literal example (see docker/process.py, Task 1) and keeps every
        # output slot the same dimensionality as a real one.
        placeholder = sitk.JoinSeries([sitk.Image(1, 1, 1, sitk.sitkFloat32)])
        sitk.WriteImage(placeholder, output_dir / "output.mha")

    return 0
