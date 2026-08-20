"""Training loop for per-control-point dose prediction.

Batch size is fixed at 1 sample per step by design: PhotonBeamEncoder isn't
vectorized across samples with different origins, and control points from
different patients in the same batch would each need their own encoder call
anyway, so batching would save nothing while adding complexity. Gradient
accumulation (via `accum_steps`) is used instead if a larger effective batch
is ever wanted.
"""
import time
from pathlib import Path

import torch

from src.data.beam_encoder import PhotonBeamEncoder


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device="cuda",
        checkpoint_dir="checkpoints",
        scheduler=None,
        grad_clip=1.0,
        log_every=20,
        amp=True,
        accum_steps=1,
        save_every_steps=2000,
        beam_encoder_cls=PhotonBeamEncoder,
    ):
        self.beam_encoder_cls = beam_encoder_cls
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.amp = amp and device == "cuda"
        self.accum_steps = accum_steps
        self.save_every_steps = save_every_steps
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.skipped_steps = 0

        # One encoder per distinct (volume_shape, voxel_spacing) -- in
        # practice just one for the whole run, since CenterCropOrPad forces
        # every patient onto the same grid. Rebuilding the coordinate grid
        # per sample would throw away the ~78x GPU speedup measured in
        # PROGRESS_LOG.md (Step 5 session); caching per-*patient* instead
        # (the original design) OOM'd a 40GB GPU on Task 3's much larger
        # grid once enough distinct patients' full-volume coordinate
        # tensors piled up in the cache (~553MB each x 67 train patients).
        # Origin is the only thing that actually varies per patient, and
        # it's now a cheap (3,) offset applied via encoder.set_origin()
        # rather than baked into the cached tensor -- see beam_encoder.py.
        self._encoder_cache = {}

    def _get_encoder(self, sample):
        volume_shape = tuple(sample["ct"].shape[-3:])
        key = (volume_shape, sample["ct_spacing"])
        encoder = self._encoder_cache.get(key)
        if encoder is None:
            encoder = self.beam_encoder_cls(volume_shape, sample["ct_spacing"], sample["ct_origin"])
            encoder.to(self.device)
            self._encoder_cache[key] = encoder
        encoder.set_origin(sample["ct_origin"])
        return encoder

    def _beam_params_to_device(self, beam_params):
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in beam_params.items()}

    def _forward(self, sample):
        ct = sample["ct"].unsqueeze(0).to(self.device)  # (1, 1, D, H, W)
        dose = sample["dose"].unsqueeze(0).to(self.device)
        body_mask = sample["body_mask"].unsqueeze(0).to(self.device)

        encoder = self._get_encoder(sample)
        beam_params = self._beam_params_to_device(sample["beam_params"])
        beam_mask = encoder.encode(beam_params, body_mask=body_mask[0]).unsqueeze(0)  # (1, 1, D, H, W)

        with torch.autocast(device_type="cuda", enabled=self.amp):
            pred = self.model(ct, beam_mask)
            loss, parts = self.loss_fn(pred, dose, body_mask)
        return loss, parts

    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        running = {}
        self.optimizer.zero_grad(set_to_none=True)
        n_logged = 0  # steps actually folded into `running` -- excludes skipped ones
        for i, sample in enumerate(self.train_loader):
            loss, parts = self._forward(sample)

            # A single degenerate sample producing inf/nan must never reach
            # backward(): once a nan gradient hits optimizer.step(), it
            # permanently corrupts every weight (the model outputs nan
            # forever after, with no recovery) -- this is exactly what
            # silently destroyed the first real training run at epoch 0
            # step ~29980, undetected for the remaining ~69 hours. See
            # PROGRESS_LOG.md.
            if not torch.isfinite(loss):
                self.skipped_steps += 1
                print(f"epoch {epoch} step {i + 1}: non-finite loss ({loss.item()}), skipping step "
                      f"(skipped_steps={self.skipped_steps})")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss / self.accum_steps).backward()

            if (i + 1) % self.accum_steps == 0:
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if not torch.isfinite(total_norm):
                        # clip_grad_norm_ can itself produce a fresh nan/inf
                        # (its own norm computation overflowing) even when
                        # the loss and GradScaler's own inf check were fine
                        # -- second line of defense, same failure mode.
                        self.skipped_steps += 1
                        print(f"epoch {epoch} step {i + 1}: non-finite grad norm ({total_norm.item()}), "
                              f"skipping optimizer step (skipped_steps={self.skipped_steps})")
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                        continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            n_logged += 1
            if (i + 1) % self.log_every == 0:
                avg = {k: v / max(n_logged, 1) for k, v in running.items()}
                print(f"epoch {epoch} step {i + 1}/{len(self.train_loader)} " +
                      " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

            if self.save_every_steps and (i + 1) % self.save_every_steps == 0:
                self.save_checkpoint(epoch, val_loss=float("nan"), is_best=False)

        if self.scheduler:
            self.scheduler.step()

        n = max(n_logged, 1)
        avg = {k: v / n for k, v in running.items()}
        print(f"[train] epoch {epoch} done in {time.time() - t0:.1f}s "
              f"(skipped {self.skipped_steps} non-finite steps total) " +
              " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        return avg

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        running = {}
        for sample in self.val_loader:
            _, parts = self._forward(sample)
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
        n = max(len(self.val_loader), 1)
        avg = {k: v / n for k, v in running.items()}
        print(f"[val] epoch {epoch} " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        return avg

    def save_checkpoint(self, epoch, val_loss, is_best):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(state, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        return state["epoch"]

    def fit(self, epochs, start_epoch=0):
        best_val = float("inf")
        for epoch in range(start_epoch, epochs):
            self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            val_loss = val_metrics.get("total", float("inf"))
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            self.save_checkpoint(epoch, val_loss, is_best)
