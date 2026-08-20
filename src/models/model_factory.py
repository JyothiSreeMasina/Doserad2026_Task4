"""Wires BeamConditionedInput + UNet3D into the full dose-prediction model."""
import torch.nn as nn

from src.models.beam_cond import BeamConditionedInput
from src.models.unet3d import build_unet3d


class DoseRADModel(nn.Module):
    def __init__(self, **unet_kwargs):
        super().__init__()
        self.condition = BeamConditionedInput()
        self.unet = build_unet3d(in_channels=self.condition.NUM_CHANNELS, out_channels=1, **unet_kwargs)

    def forward(self, ct, beam_mask):
        x = self.condition(ct, beam_mask)
        return self.unet(x)


def build_model(config=None):
    config = config or {}
    return DoseRADModel(**config)
