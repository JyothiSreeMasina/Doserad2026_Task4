"""Preprocessing transforms for DoseRAD2026 samples.

Each transform takes and returns the sample dict produced by
`DoseRAD2026Dataset.__getitem__` (`ct`, `dose`, `ct_spacing`, `ct_origin`,
`beam_params`, ...). Intended composition order:

    Compose([ResampleToSpacing(), CenterCropOrPad(), BodyMask(), NormalizeCT(), RandomIntensityShift()])

Note BodyMask must run before NormalizeCT -- it thresholds raw HU values,
and NormalizeCT rescales them to [0, 1].

Geometric augmentation (flip/rotate) is deliberately NOT implemented here:
flipping the CT/dose array without also transforming `beam_params`
(gantry_angle, iso_center, MLC positions -- all in real patient physical
coordinates) would silently decouple the input geometry from the target
dose. Only intensity-only augmentation is safe without that extra geometry
bookkeeping, so that's what's provided.
"""
import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ResampleToSpacing:
    """Resamples ct/dose to a target voxel spacing (mm), trilinear.

    All patients inspected so far already ship at (2,2,2)mm (the challenge's
    own release pipeline pre-resamples photon-task data to this grid -- see
    PROGRESS_LOG.md), so this is a no-op in practice and untested against a
    real mismatched-spacing case. Kept as a safety net rather than an
    assumption, in case a patient in the full 77 differs.
    """

    def __init__(self, target_spacing=(2.0, 2.0, 2.0), tol=1e-3):
        self.target_spacing = target_spacing
        self.tol = tol

    def __call__(self, sample):
        spacing = sample["ct_spacing"]
        if all(abs(s - t) < self.tol for s, t in zip(spacing, self.target_spacing)):
            return sample

        scale = [s / t for s, t in zip(spacing, self.target_spacing)]
        for key in ("ct", "dose"):
            vol = sample[key].unsqueeze(0)  # (1, C, D, H, W)
            new_shape = [max(1, round(vol.shape[2 + i] * scale[i])) for i in range(3)]
            sample[key] = F.interpolate(vol, size=new_shape, mode="trilinear", align_corners=False).squeeze(0)
        sample["ct_spacing"] = self.target_spacing
        return sample


class CenterCropOrPad:
    """Crops or pads ct/dose (center-aligned) to a fixed shape for batching.

    Patient volumes vary in size (e.g. 1ABB006 is 246x246x249 vs 1ABB011's
    141x246x248, in z,y,x) even though spacing is uniform, so DataLoader
    can't stack them as-is. Updates `ct_origin` to match, so the physical
    (mm) position of voxel (0,0,0) stays correct after the crop/pad -- this
    matters for Step 4's beam ray-tracing, which needs to map iso_center
    (physical mm) back into array indices.
    """

    def __init__(self, target_shape=(256, 256, 256), ct_pad_value=-1024.0, dose_pad_value=0.0):
        self.target_shape = target_shape
        self.ct_pad_value = ct_pad_value
        self.dose_pad_value = dose_pad_value

    def _crop_or_pad(self, vol, pad_value):
        # vol: (C, D, H, W)
        offsets = []
        for d in range(3):
            in_size = vol.shape[1 + d]
            target = self.target_shape[d]
            offsets.append((in_size - target) // 2)  # >0 => crop start; <0 => -pad_before

        slices = [slice(None)]
        pad = []  # F.pad wants (W_before, W_after, H_before, H_after, D_before, D_after)
        for d in range(3):
            in_size = vol.shape[1 + d]
            target = self.target_shape[d]
            off = offsets[d]
            if off >= 0:
                slices.append(slice(off, off + target))
            else:
                slices.append(slice(0, in_size))
        vol = vol[tuple(slices)]

        for d in reversed(range(3)):
            in_size = vol.shape[1 + d]
            target = self.target_shape[d]
            deficit = target - in_size
            before = deficit // 2
            after = deficit - before
            pad.extend([max(before, 0), max(after, 0)])
        vol = F.pad(vol, pad, mode="constant", value=pad_value)
        return vol, offsets

    def restore(self, vol, original_shape):
        """Inverse of `_crop_or_pad`: maps a (C, *target_shape) volume back
        to (C, *original_shape) under the same center-alignment convention.

        Used at inference time (docker/process.py) to put a predicted dose
        volume back onto a patient's native grid before writing output --
        much faster than a general sitk.Resample when spacing is unchanged
        (the common case for all real data seen so far, see PROGRESS_LOG.md),
        since it's a pure GPU tensor op with no CPU round-trip. Regions that
        were cropped away in the forward direction (never seen by the model)
        are filled with zero -- a sound default, since there's no prediction
        to put there.
        """
        # Mirrors `_crop_or_pad`'s exact per-branch arithmetic (not a single
        # negated offset) -- for an odd size difference, floor-dividing a
        # positive deficit and floor-dividing its negation give different
        # results in Python (e.g. 7//2=3 but -(-7//2)=4), so deriving the
        # inverse from one signed `offset` silently mismatched forward's
        # actual `before` padding by a voxel on any odd-difference axis.
        # Caught by round-trip-testing this against real patient shapes
        # before using it in docker/process.py -- see PROGRESS_LOG.md.
        slices = [slice(None)]
        for d in range(3):
            in_size = original_shape[d]
            target = self.target_shape[d]
            if in_size >= target:
                slices.append(slice(None))  # forward cropped -> vol is already target-sized here
            else:
                deficit = target - in_size
                before = deficit // 2
                slices.append(slice(before, before + in_size))  # undo the forward pad
        vol = vol[tuple(slices)]

        pad = []
        for d in reversed(range(3)):
            in_size = original_shape[d]
            target = self.target_shape[d]
            if in_size >= target:
                before = (in_size - target) // 2  # forward's crop start
                after = in_size - (vol.shape[1 + d] + before)
                pad.extend([max(before, 0), max(after, 0)])
            else:
                pad.extend([0, 0])
        return F.pad(vol, pad, mode="constant", value=0.0)

    def __call__(self, sample):
        sample["ct"], offsets = self._crop_or_pad(sample["ct"], self.ct_pad_value)
        # "dose" is absent at inference time (docker/process.py has no
        # ground truth to crop, only a CT to predict from) -- only touch it
        # when present, e.g. during training/evaluation.
        if "dose" in sample:
            sample["dose"], _ = self._crop_or_pad(sample["dose"], self.dose_pad_value)

        # offsets are in (z,y,x) order; ct_origin is in sitk (x,y,z) order.
        spacing = sample["ct_spacing"]
        origin = list(sample["ct_origin"])
        axis_map = {0: 2, 1: 1, 2: 0}  # d (z,y,x) -> origin index (x,y,z)
        for d in range(3):
            origin[axis_map[d]] += offsets[d] * spacing[d]
        sample["ct_origin"] = tuple(origin)

        return sample


class NormalizeCT:
    """Clips CT to an HU window and min-max scales to [0, 1]."""

    def __init__(self, hu_min=-1000.0, hu_max=3000.0):
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __call__(self, sample):
        ct = torch.clamp(sample["ct"], self.hu_min, self.hu_max)
        sample["ct"] = (ct - self.hu_min) / (self.hu_max - self.hu_min)
        return sample


class ScaleDose:
    """Multiplies dose by a fixed constant so targets sit near O(1) instead
    of the raw ~5e-5 to 1e-4 Gy per-control-point scale confirmed on real
    data (see PROGRESS_LOG.md). Without this, a freshly-initialized model's
    unconstrained output (order 0.1-1, no final activation) is ~1e4-1e5x
    larger than the target, which produced huge, fp16-autocast-overflowing
    per-sample losses and permanently corrupted training weights with NaN
    partway through the very first epoch of the first real run.

    All the official metrics (Masked MAE, IDD curve distance, Gamma Index,
    Stratified Plan MAE) are relative/normalized, so this scale factor is
    invisible to them as long as pred and target are compared in the same
    units. Inference/submission code MUST divide the raw model output by
    this same factor before writing out final dose predictions in Gy.
    """

    def __init__(self, scale=10000.0):
        self.scale = scale

    def __call__(self, sample):
        sample["dose"] = sample["dose"] * self.scale
        return sample


class BodyMask:
    """Simple HU-threshold body mask (air vs. tissue), not a full segmentation.

    Must run before NormalizeCT (operates on raw HU values).
    """

    def __init__(self, hu_threshold=-500.0):
        self.hu_threshold = hu_threshold

    def __call__(self, sample):
        sample["body_mask"] = (sample["ct"] > self.hu_threshold).float()
        return sample


class BodyMaskMR:
    """MR body mask: simple nonzero-intensity threshold, not a full segmentation.

    Unlike CT, MR intensities aren't physically calibrated (no HU equivalent),
    so a fixed HU-style threshold doesn't transfer. Empirically checked
    instead (2026-08-15, 5 patients): 0.35T bSSFP background/air is reliably
    ~0 (median MR intensity across every patient sampled was exactly 0.0),
    so `mr > 1.0` already recovers a mask matching the CT-derived body mask
    at ~0.87 mean IoU (range 0.77-0.99) -- a plain Otsu threshold on the same
    data actually did worse (~0.68 mean IoU), because the background spike
    at exactly 0 skews Otsu's bimodal split away from the true air/tissue
    boundary. Must run before NormalizeMR (operates on raw intensities).
    """

    def __init__(self, mr_threshold=1.0):
        self.mr_threshold = mr_threshold

    def __call__(self, sample):
        sample["body_mask"] = (sample["ct"] > self.mr_threshold).float()
        return sample


class NormalizeMR:
    """Per-volume percentile clip + min-max scale to [0, 1].

    MR intensities have no cross-patient physical calibration (unlike CT's
    HU), so a fixed clip window like NormalizeCT's would be wrong: the 99th
    percentile intensity measured directly across 8 real patients ranged
    ~170 to ~409, and the raw max ranged ~369 to ~1224 (2026-08-15 check) --
    over 3x variation in scale alone. Normalizing per-volume against its own
    percentile instead makes every sample comparable regardless of that
    per-scan scale drift.
    """

    def __init__(self, clip_percentile=99.0):
        self.clip_percentile = clip_percentile

    def __call__(self, sample):
        mr = sample["ct"]
        # torch.quantile has a hard 2^24-element cap (errors with "input
        # tensor is too large") -- fine for Task 2's photon grid (256^3 ~=
        # 16.7M) but Task 4's proton grid (176x512x512 ~= 46.1M) blows past
        # it. np.percentile has no such limit and this already runs
        # CPU-side in a DataLoader worker, so there's no extra device
        # transfer cost to switching.
        hi = float(np.percentile(mr.numpy(), self.clip_percentile))
        hi = max(hi, 1e-6)  # guard a degenerate all-zero volume
        mr = torch.clamp(mr, 0.0, hi)
        sample["ct"] = mr / hi
        return sample


class RandomIntensityShift:
    """Additive Gaussian noise on (already-normalized) CT intensities.

    Intensity-only augmentation is safe because it doesn't move anything in
    physical space, so beam_params stay valid without adjustment.
    """

    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, sample):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(sample["ct"]) * self.std
            sample["ct"] = sample["ct"] + noise
        return sample
