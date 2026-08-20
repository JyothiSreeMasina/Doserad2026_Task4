"""Loss functions for dose prediction, aligned with the challenge's official metric.

Reference (DoseRAD2026_Complete_Reference.md): "Masked MAE: MAE in the
high-dose region (>=10% of beam max), normalized by beam max." Training
optimizes a differentiable version of exactly that (MaskedMAELoss), plus a
small body-masked full-field term (BodyMaskedL1Loss) so the network still
gets gradient signal in the low/mid-dose region -- without it, everything
outside the high-dose mask is free to be arbitrary, which would score fine
on Masked MAE alone but hurt the other official metrics that look at the
full distribution (Gamma Index, DVH-based Clinical Score).

2026-08-10: the first full training run (see PROGRESS_LOG.md) converged to
a model that dumps a large fraction of predicted dose mass outside the
patient body -- masked_mae looked fine (that metric never looks outside
the high-dose region) but idd_curve_distance and gamma_index_pass_rate on
the full evaluation suite were badly hurt by it. Root cause: neither term
above gave the network any gradient signal to suppress dose outside the
body -- BodyMaskedL1Loss only ever looks *inside* body_mask. Verified
directly against real data that GT dose outside the body is genuinely
near-zero on average (~1e-8, five to six orders of magnitude below beam
max, versus ~1e-4 in-body) before adding OutsideBodyL1Loss below to
penalize it explicitly.
"""
import torch
import torch.nn as nn


class MaskedMAELoss(nn.Module):
    """Differentiable version of the official Masked MAE metric.

    Per sample: mask = dose >= high_dose_frac * dose.max(); loss =
    mean(|pred - dose| within mask) / dose.max(). Averaged over the batch.
    """

    def __init__(self, high_dose_frac=0.1, eps=1e-8):
        super().__init__()
        self.high_dose_frac = high_dose_frac
        self.eps = eps

    def forward(self, pred, target):
        losses = []
        for p, t in zip(pred, target):
            beam_max = t.max()
            if beam_max < self.eps:
                continue
            mask = t >= self.high_dose_frac * beam_max
            if mask.sum() == 0:
                continue
            losses.append((p[mask] - t[mask]).abs().mean() / beam_max)
        if not losses:
            return pred.sum() * 0.0  # keeps autograd graph alive on a degenerate all-zero batch
        return torch.stack(losses).mean()


class BodyMaskedL1Loss(nn.Module):
    """Full-field L1 within the patient body, normalized per-sample by beam max.

    Auxiliary term: supplies gradient signal in the low/mid-dose region that
    MaskedMAELoss ignores entirely.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, body_mask):
        losses = []
        for p, t, m in zip(pred, target, body_mask):
            beam_max = t.max()
            body = m > 0.5
            if beam_max < self.eps or body.sum() == 0:
                continue
            losses.append((p[body] - t[body]).abs().mean() / beam_max)
        if not losses:
            return pred.sum() * 0.0
        return torch.stack(losses).mean()


class OutsideBodyL1Loss(nn.Module):
    """Full-field L1 outside the patient body, normalized per-sample by beam max.

    GT dose outside the body is verified near-zero on real data (~1e-8,
    5-6 orders of magnitude below beam max -- see module docstring), so
    this directly penalizes the network for spreading predicted dose mass
    into air/background, which MaskedMAELoss and BodyMaskedL1Loss both
    ignore entirely (they only ever look inside their respective masks).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, body_mask):
        losses = []
        for p, t, m in zip(pred, target, body_mask):
            beam_max = t.max()
            outside = m <= 0.5
            if beam_max < self.eps or outside.sum() == 0:
                continue
            losses.append((p[outside] - t[outside]).abs().mean() / beam_max)
        if not losses:
            return pred.sum() * 0.0
        return torch.stack(losses).mean()


class DoseLoss(nn.Module):
    """Combined training loss: metric-aligned high-dose term + auxiliary
    body and outside-body terms.

    Weights default to favoring the metric-aligned term, since that's what's
    actually scored; the body and outside-body terms are smaller regularizers.
    """

    def __init__(self, high_dose_frac=0.1, w_masked=1.0, w_body=0.2, w_outside=0.2):
        super().__init__()
        self.masked_mae = MaskedMAELoss(high_dose_frac=high_dose_frac)
        self.body_l1 = BodyMaskedL1Loss()
        self.outside_l1 = OutsideBodyL1Loss()
        self.w_masked = w_masked
        self.w_body = w_body
        self.w_outside = w_outside

    def forward(self, pred, target, body_mask):
        masked = self.masked_mae(pred, target)
        body = self.body_l1(pred, target, body_mask)
        outside = self.outside_l1(pred, target, body_mask)
        total = self.w_masked * masked + self.w_body * body + self.w_outside * outside
        parts = {
            "masked_mae": masked.item(), "body_l1": body.item(),
            "outside_l1": outside.item(), "total": total.item(),
        }
        return total, parts
