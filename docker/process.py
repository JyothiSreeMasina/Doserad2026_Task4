"""DoseRAD2026 Task 1 (photon/CT) inference for Grand Challenge's `invoke` API.

Reads the batched input contract at /input (up to 10 source CT images plus
one beam-level metadata JSON covering all of them), runs one model forward
pass per control point (the unit the challenge times against the 1s/beam
limit -- see DoseRAD2026_Complete_Reference.md's "Photon: per control point"
note), and writes the ten stacked-dose-map output slots at /output.

Preprocessing (resample/crop-pad/body-mask/normalize) reuses the exact same
`src/data/transforms.py` classes used during training -- critical, since any
drift between train-time and inference-time preprocessing would silently
invalidate the model's predictions. The one extra step needed only at
inference time: our internal grid (256^3 @ 2mm, center-cropped/padded) does
not match the original input image's native grid, but the challenge requires
"Output dose grid must match input image exactly (identical size, spacing,
origin, direction)". That's handled by building a proper SimpleITK image on
our internal grid (using the origin/spacing our transform pipeline already
tracks) and calling sitk.Resample with the *original* CT image as the
reference grid -- this correctly undoes both the crop/pad and any resample
in one physically-grounded operation, rather than hand-rolling inverse array
slicing. Voxels outside our modeled 256^3 window (if a patient's native FOV
is larger) default to zero dose.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from src.data.beam_encoder import PhotonBeamEncoder
from src.data.transforms import BodyMask, CenterCropOrPad, Compose, NormalizeCT, ResampleToSpacing
from src.models.model_factory import build_model

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
CONFIG_PATH = Path("/opt/app/configs/task1_photon_ct.yaml")
MODEL_DIR = Path("/opt/ml/model")

NUM_OUTPUT_FILES = 10
CT_DIR_BASE = "radiation-dose-calculation-source-ct-image"
METADATA_JSON_NAME = "stacked-photon-beam-level-metadata"


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
    the first real /invoke call. Measured ~0.1s/CP steady-state vs several
    seconds for a cold first call -- if a real evaluation invokes with as
    few as one beam per call, that cold-start cost alone could blow the
    1s/beam limit on the very first scored beam. Exercises the *exact* same
    path as a real invoke -- both the model and PhotonBeamEncoder.encode(),
    not just the model -- since encode() does its own per-call tensor ops
    (rotation, meshgrid-derived masking) that pay their own first-call cost
    if left unwarmed.
    """
    dcfg = cfg["data"]
    target_shape = tuple(dcfg["target_shape"])
    target_spacing = tuple(dcfg["target_spacing"])
    amp = cfg["optim"].get("amp", True)

    dummy_ct = torch.zeros((1, 1, *target_shape), device=device)
    encoder = PhotonBeamEncoder(target_shape, target_spacing, origin=(0.0, 0.0, 0.0))
    encoder.to(device)
    dummy_beam_params = {
        "gantry_angle": 0.0,
        "iso_center": torch.zeros(3, device=device),
        "SAD": 1000.0,
        "mlc_left_int_mm": torch.full((PhotonBeamEncoder.NUM_LEAVES,), -20.0, device=device),
        "mlc_right_int_mm": torch.full((PhotonBeamEncoder.NUM_LEAVES,), 20.0, device=device),
    }

    with torch.no_grad():
        for _ in range(3):
            beam_mask = encoder.encode(dummy_beam_params).unsqueeze(0)
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
    padded, possibly resampled) grid. Returns an sitk.Image on
    `original_image`'s exact grid (size, spacing, origin, direction), with
    `minimum_cutoff` applied.

    Fast path: when spacing is unchanged (ResampleToSpacing was a no-op --
    true for every real patient checked so far, see PROGRESS_LOG.md), the
    crop/pad is a pure center-alignment op, so `CenterCropOrPad.restore()`
    inverts it exactly with a GPU tensor op -- no CPU round-trip. Falls back
    to sitk.Resample (slower, CPU-bound, but fully general) only if spacing
    genuinely differs -- kept for robustness against an unknown
    hidden-test-set patient, not expected to trigger.

    The cutoff is applied here, on the GPU, before the *one* necessary
    CPU transfer -- profiling showed a second full sitk
    GetArrayFromImage/GetImageFromArray round trip (previously done in
    run() just to apply this threshold) cost ~80ms/CP on its own, and an
    `.astype(np.float32)` on an already-float32 array cost another ~40ms/CP
    from an unnecessary extra copy (`ndarray.astype` copies by default even
    when the dtype already matches). Both were pure waste: cut per-CP time
    from ~0.33s to a fraction of that. See PROGRESS_LOG.md.
    """
    dose_tensor = torch.where(dose_tensor < minimum_cutoff, torch.zeros_like(dose_tensor), dose_tensor)
    original_shape_zyx = tuple(reversed(original_image.GetSize()))
    spacing_matches = tuple(round(s, 6) for s in sample["ct_spacing"]) == \
        tuple(round(s, 6) for s in reversed(original_image.GetSpacing()))

    if spacing_matches:
        restored = restore_cropper.restore(dose_tensor.unsqueeze(0), original_shape_zyx).squeeze(0)
        dose_img = sitk.GetImageFromArray(restored.cpu().numpy())
        dose_img.CopyInformation(original_image)
        return dose_img

    dose_img = sitk.GetImageFromArray(dose_tensor.cpu().numpy())
    dose_img.SetSpacing(tuple(reversed(sample["ct_spacing"])))
    dose_img.SetOrigin(sample["ct_origin"])
    dose_img.SetDirection(original_image.GetDirection())
    return sitk.Resample(
        dose_img,
        referenceImage=original_image,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        outputPixelType=sitk.sitkFloat32,
    )


@torch.no_grad()
def predict_control_point(model, encoder, ct_tensor, beam, cp, device, dose_scale, amp):
    beam_params = {
        "gantry_angle": float(cp["gantry_angle"]),
        "iso_center": torch.tensor(beam["iso_center"], dtype=torch.float32, device=device),
        "SAD": float(beam["SAD"]),
        "mlc_left_int_mm": torch.tensor(cp["mlc_left_int_mm"], dtype=torch.float32, device=device),
        "mlc_right_int_mm": torch.tensor(cp["mlc_right_int_mm"], dtype=torch.float32, device=device),
    }
    beam_mask = encoder.encode(beam_params).unsqueeze(0)
    with torch.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
        pred = model(ct_tensor, beam_mask)
    return (pred.squeeze(0).squeeze(0) / dose_scale).float()  # (D, H, W), stays on device


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
    per_output = [dict() for _ in range(NUM_OUTPUT_FILES)]  # idx_in_output -> sitk.Image

    for image_entry in metadata:
        image_idx = image_entry["image_file_idx"]
        location = INPUT_PATH / f"images/{CT_DIR_BASE}-{image_idx + 1}"
        original_image = load_sitk_image(location)
        print(f"Image {image_idx + 1}: size={original_image.GetSize()} spacing={original_image.GetSpacing()}")

        sample = preprocess_ct(original_image, target_spacing, target_shape)
        ct_tensor = sample["ct"].unsqueeze(0).to(device)

        volume_shape = tuple(sample["ct"].shape[-3:])
        encoder = PhotonBeamEncoder(volume_shape, sample["ct_spacing"], sample["ct_origin"])
        encoder.to(device)

        n_beams = len(image_entry["beams"])
        for beam in image_entry["beams"]:
            n_cps = len(beam["control_points"])
            print(f"  Beam: {n_cps} control points")
            for cp in beam["control_points"]:
                output_info = cp["output_info"]
                minimum_cutoff = float(output_info["minimum_cutoff"])
                dose_tensor = predict_control_point(model, encoder, ct_tensor, beam, cp, device, dose_scale, amp)
                dose_img = restore_dose_to_original_grid(
                    dose_tensor, minimum_cutoff, restore_cropper, sample, original_image
                )
                per_output[output_info["output_file_idx"]][output_info["idx_in_output"]] = dose_img
        print(f"Image {image_idx + 1}: {n_beams} beam(s) done")

    for output_index in range(NUM_OUTPUT_FILES):
        slot = per_output[output_index]
        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_index + 1}"
        os.makedirs(output_dir, exist_ok=True)

        if not slot:
            # Genuine 4D (1,1,1,1) placeholder -- matches the instructions' literal
            # example and keeps every output slot the same dimensionality as a real
            # one (always built via JoinSeries below). A bare sitk.Image(1, 1, ...)
            # would silently be a 2D (1,1) image instead (SimpleITK's 2-size-arg
            # constructor), not the 1x1x1x1 the spec shows.
            placeholder = sitk.JoinSeries([sitk.Image(1, 1, 1, sitk.sitkFloat32)])
            sitk.WriteImage(placeholder, output_dir / "output.mha")
            continue

        stack_size = max(slot) + 1
        dose_slices = [slot[i] for i in range(stack_size)]
        stacked = sitk.JoinSeries(dose_slices)
        sitk.WriteImage(stacked, output_dir / "output.mha", useCompression=False)
        print(f"Wrote output slot {output_index + 1}: {stack_size} slice(s)")

    return 0
