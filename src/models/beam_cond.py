"""Combines CT and the beam-encoder aperture mask into the model's input tensor."""
import torch


class BeamConditionedInput:
    """Channels: [normalized CT, beam aperture mask].

    body_mask (from src/data/transforms.py) is deliberately NOT included as
    a model input -- it's used for loss masking (src/training/losses.py)
    instead, since air-vs-tissue is already directly visible in the CT
    channel and wouldn't add information for the network itself.
    """

    NUM_CHANNELS = 2

    def __call__(self, ct, beam_mask):
        # dim=-4 is the channel dim whether or not a batch dim is present:
        # (C,D,H,W) -> dim 0, or (B,C,D,H,W) -> dim 1.
        return torch.cat([ct, beam_mask], dim=-4)
