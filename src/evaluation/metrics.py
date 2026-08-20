"""Official DoseRAD2026 evaluation metrics (per DoseRAD2026_Complete_Reference.md).

Level 1 (per beam/control point) -- fully computable from the local training
data:
  - masked_mae: "MAE in the high-dose region (>=10% of beam max), normalized
    by beam max". Same formula as the training loss in src/training/losses.py
    (that loss is a differentiable copy of this metric), reimplemented here
    without autograd for standalone evaluation.
  - idd_curve_distance: "RMSD of Integrated Depth-Dose curves, normalized by
    peak GT IDD". Depth is measured along the control point's central beam
    axis, reusing PhotonBeamEncoder.beam_depth() (src/data/beam_encoder.py)
    so the geometry matches exactly what the model was conditioned on.

Level 2 (full-plan, aggregated across a patient's beams) -- KNOWN GAP:
DoseRAD2026's local `photon/training/<PID>/` folders ship only `image/`
(ct.mha, mr.mha), `dose/Dose_B{beam}_CP{cp}.mha`, and `<PID>.json` (beam
geometry). There is no prescription-dose field and no RTSTRUCT/segmentation
data anywhere in the download (verified across the full 77-patient set, not
just a sample -- see PROGRESS_LOG.md). That means:
  - stratified_plan_mae and gamma_index_3d need a "prescription" dose to
    define their evaluation region/strata; the reference doc never states
    where it comes from for local use, so both accept it as a parameter and
    fall back to the plan's own max dose as a documented approximation if
    the caller doesn't supply one. This is a real limitation, not just a
    style choice -- a true prescription can shift strata boundaries and the
    gamma evaluation region.
  - dvh_clinical_score needs real PTV/OAR contours. Fabricating placeholder
    contours would produce a number that looks like a real metric but means
    nothing, so this function requires the caller to supply `structures`
    and returns None (not a fabricated score) if none are given. If the
    official Grand Challenge evaluation container supplies structures at
    submission time, this function is ready to use them as-is.
"""
import numpy as np
import torch
import torch.nn.functional as F


def masked_mae(pred, target, high_dose_frac=0.1, eps=1e-8):
    """Official Masked MAE for one beam/control point. pred, target: (D,H,W) or (1,D,H,W)."""
    pred = pred.squeeze()
    target = target.squeeze()
    beam_max = target.max()
    if beam_max < eps:
        return float("nan")
    mask = target >= high_dose_frac * beam_max
    if mask.sum() == 0:
        return float("nan")
    return ((pred[mask] - target[mask]).abs().mean() / beam_max).item()


def idd_curve_distance(pred, target, encoder, beam_params, bin_width_mm=2.0, eps=1e-8):
    """Official IDD Curve Distance for one control point.

    Bins dose by depth along the beam's central axis (collapsing the
    transverse plane at each depth), then RMSD between the predicted and GT
    depth-dose curves, normalized by the GT curve's peak.
    """
    depth = encoder.beam_depth(beam_params).flatten()
    pred_flat = pred.squeeze().flatten()
    target_flat = target.squeeze().flatten()

    valid = depth > 0  # voxels "behind" the source aren't part of the beam path
    depth, pred_flat, target_flat = depth[valid], pred_flat[valid], target_flat[valid]
    if depth.numel() == 0:
        return float("nan")

    bin_idx = torch.clamp((depth / bin_width_mm).long(), min=0)
    n_bins = int(bin_idx.max().item()) + 1
    pred_idd = torch.zeros(n_bins, device=depth.device).scatter_add_(0, bin_idx, pred_flat)
    target_idd = torch.zeros(n_bins, device=depth.device).scatter_add_(0, bin_idx, target_flat)

    peak = target_idd.max()
    if peak < eps:
        return float("nan")
    rmsd = torch.sqrt(((pred_idd - target_idd) ** 2).mean())
    return (rmsd / peak).item()


def stratified_plan_mae(pred_plan, target_plan, prescription=None, eps=1e-8):
    """Official Stratified Plan MAE, computed on a full-plan dose (sum of all
    of a patient's per-control-point doses across all beams -- see
    src/evaluation/evaluate.py for how that sum is built).

    prescription: see module docstring -- not available locally; defaults to
    target_plan.max() if not supplied.
    """
    pred_plan = pred_plan.squeeze()
    target_plan = target_plan.squeeze()
    prescription = float(prescription) if prescription is not None else target_plan.max().item()
    if prescription < eps:
        return {"high": float("nan"), "mid": float("nan"), "low": float("nan"), "mean": float("nan")}

    strata = {
        "high": (0.8 * prescription, float("inf")),
        "mid": (0.3 * prescription, 0.8 * prescription),
        "low": (0.1 * prescription, 0.3 * prescription),
    }
    diff = (pred_plan - target_plan).abs()
    result = {}
    for name, (lo, hi) in strata.items():
        mask = (target_plan >= lo) & (target_plan < hi)
        result[name] = (diff[mask].mean() / prescription).item() if mask.sum() > 0 else float("nan")

    valid = [v for v in result.values() if not np.isnan(v)]
    result["mean"] = float(np.mean(valid)) if valid else float("nan")
    return result


def gamma_index_3d(pred, target, spacing, dose_percent=1.0, distance_mm=1.0, prescription=None, threshold_frac=0.1, eps=1e-8):
    """Official 3D Local Gamma Index (1%/1mm by default).

    'Local' normalization: the dose-difference term at each evaluated voxel
    is normalized by that voxel's own GT dose (not a single global
    reference) -- per the reference doc's "local normalization" wording.
    Searches a voxel window sized to `distance_mm` around each point and
    takes the minimum gamma value over that window, which is the standard
    gamma-index formulation.

    prescription: only used to pick the evaluation region (>= threshold_frac
    * prescription) -- see module docstring for the same local-data gap as
    stratified_plan_mae. Defaults to this volume's own max dose.

    spacing: (dz, dy, dx) mm, matching this codebase's array-axis convention.
    """
    pred = pred.squeeze()
    target = target.squeeze()
    prescription = float(prescription) if prescription is not None else target.max().item()
    if prescription < eps:
        return float("nan")

    eval_mask = target >= threshold_frac * prescription
    if eval_mask.sum() == 0:
        return float("nan")

    radius_vox = [max(1, int(np.ceil(distance_mm / s))) for s in spacing]
    pad = [radius_vox[2], radius_vox[2], radius_vox[1], radius_vox[1], radius_vox[0], radius_vox[0]]
    big = target.max().item() * 1e6 + 1.0  # sentinel: guarantees gamma > 1 for out-of-bounds shifts
    pred_pad = F.pad(pred[None, None], pad, mode="constant", value=big).squeeze(0).squeeze(0)

    dz, dy, dx = spacing
    dose_crit = (dose_percent / 100.0) * target.clamp(min=eps)
    gamma_min = torch.full_like(target, float("inf"))
    for oz in range(-radius_vox[0], radius_vox[0] + 1):
        for oy in range(-radius_vox[1], radius_vox[1] + 1):
            for ox in range(-radius_vox[2], radius_vox[2] + 1):
                dist = (oz * dz) ** 2 + (oy * dy) ** 2 + (ox * dx) ** 2
                dist = dist ** 0.5
                shifted = pred_pad[
                    radius_vox[0] + oz: radius_vox[0] + oz + target.shape[0],
                    radius_vox[1] + oy: radius_vox[1] + oy + target.shape[1],
                    radius_vox[2] + ox: radius_vox[2] + ox + target.shape[2],
                ]
                dose_term = (target - shifted) / dose_crit
                dist_term = dist / distance_mm
                candidate = torch.sqrt(dose_term ** 2 + dist_term ** 2)
                gamma_min = torch.minimum(gamma_min, candidate)

    passed = (gamma_min[eval_mask] <= 1.0).float().mean()
    return passed.item()


def _dvh_point_stats(dose, mask, d_levels):
    values = dose[mask.squeeze().bool()]
    if values.numel() == 0:
        return {}
    sorted_vals, _ = torch.sort(values, descending=True)
    n = sorted_vals.numel()
    out = {f"D{x}": sorted_vals[min(int(round(x / 100.0 * n)), n - 1)].item() for x in d_levels}
    out["Dmean"] = values.mean().item()
    return out


def dvh_clinical_score(pred_plan, target_plan, structures, prescription, eps=1e-8):
    """Official DVH-Based Clinical Score: PTV D98%/V95%, 3 closest OARs'
    D2%/Dmean, averaged relative difference between predicted and GT.

    structures: dict of {"PTV": mask, oar_name: mask, ...} binary tensors,
    same shape as pred_plan/target_plan. See module docstring -- this data
    isn't present in the local DoseRAD2026 training set, so returns None
    (does not fabricate a score) when structures isn't supplied.
    """
    if not structures:
        return None

    pred_plan = pred_plan.squeeze()
    target_plan = target_plan.squeeze()
    diffs = []
    for name, mask in structures.items():
        d_levels = (98, 2) if name == "PTV" else (2,)
        pred_stats = _dvh_point_stats(pred_plan, mask, d_levels)
        gt_stats = _dvh_point_stats(target_plan, mask, d_levels)
        if not gt_stats:
            continue

        if name == "PTV":
            m = mask.squeeze().bool()
            pred_stats["V95"] = (pred_plan[m] >= 0.95 * prescription).float().mean().item()
            gt_stats["V95"] = (target_plan[m] >= 0.95 * prescription).float().mean().item()

        for key, gt_val in gt_stats.items():
            if abs(gt_val) < eps:
                continue
            diffs.append(abs(pred_stats[key] - gt_val) / abs(gt_val))

    return float(np.mean(diffs)) if diffs else float("nan")
