# DoseRAD2026 — Progress Log

> Running minutes of every session: what was checked, what was decided, and exactly what changed on disk. Newest session on top. Cross-reference `CHECKLIST.md` for the overall task list this log tracks progress against.

---

## Session: 2026-08-12/14 — Fixed outside-body dose leakage, fine-tuned, repackaged Docker for upload

### Steps taken, in order

1. Fetched the real `submission-instructions` page from the challenge site and cross-checked our container against every stated requirement (invoke label, health/invoke endpoints, `JoinSeries` for 4D stacking, all-10-slots-always contract, exact output grid matching, minimum-cutoff zeroing) — already compliant on all points, no code changes needed there.
2. Read `src/training/losses.py`/`trainer.py`, confirmed the root-cause hypothesis from the 2026-08-10 entry structurally: no loss term penalized dose predicted outside the body mask.
3. Empirically verified (ad hoc check, 5 real validation samples) that GT dose outside the body is genuinely near-zero (~1e-8) vs. ~7e-5 in-body dose max, before trusting the fix.
4. Added `OutsideBodyL1Loss` to `src/training/losses.py` and wired it into `DoseLoss` as a third weighted term (`w_outside=0.2`); added the matching `w_outside: 0.2` entry to `configs/task1_photon_ct.yaml`'s `loss:` block.
5. Backed up the pre-fix checkpoints (`checkpoints/task1_photon_ct/best_pre_outside_body_fix_epoch4.pt`, `last_pre_outside_body_fix_epoch9.pt`) before touching anything.
6. Given the shrinking runway (deadline discovered to be 2026-08-15, not the 5-day margin assumed earlier), chose to **resume/fine-tune from `best.pt` (epoch 4)** with the fixed loss rather than retrain from scratch (~25hr) — launched `scripts/train.py --resume ... best_pre_outside_body_fix_epoch4.pt`, running epochs 5-9 (5 more epochs, ~2.7hr/epoch).
7. After epoch 5 (the first fix epoch) finished, ran the full metric suite on it *before* waiting for the rest of the run, to validate the fix early: **`idd_curve_distance` dropped from 126.5 to 0.56** (~225x), `gamma_index_pass_rate` rose from 1.3% to 3.3%, `masked_mae` unchanged (9.5%) — confirms the root-cause diagnosis was correct.
8. Training ran to completion (all 10 epochs, no crashes; NaN-guard skip rate stayed low, 79/36180 by epoch 9). Val loss was actually lowest at epoch 5 among epochs 5-9 (later epochs mildly overfit again), so `best.pt` correctly still points to epoch 5 — no further action needed there.
9. Swapped the epoch-5 checkpoint into `docker/model/task1_photon_ct/last.pt`, rebuilt the container, and re-ran `do_test_run.sh` — confirmed correct checkpoint load (`"Loaded checkpoint ... (epoch 5)"`), `/invoke` succeeded in 4.72s for 4 control points, correct output.
10. Ran `docker/do_save.sh` — produced the final upload artifacts: `docker/doserad2026_task1_photon_ct_2026-08-14T10-18-36.670007874-04-00.tar.gz` (3.7G, the image) and `docker/model.tar.gz` (51M, the weights, uploaded separately per Grand Challenge convention).

### Findings — fix confirmed effective, deadline now critical

| Metric | Pre-fix (epoch 4) | Post-fix (epoch 5) |
|---|---|---|
| `masked_mae` | 0.0953 | 0.0949 |
| `idd_curve_distance` | 126.5 | **0.56** |
| `stratified_plan_mae` | 0.0716 | 0.0509 |
| `gamma_index_pass_rate` | 0.0135 (1.3%) | 0.0328 (3.3%) |

`idd_curve_distance` is now solidly in the expected O(0.01-1) range — the fix worked. `gamma_index_pass_rate` is still low; unclear how much of this is genuine model inaccuracy vs. an artifact of the local prescription-stand-in hack (no real RTSTRUCT data available locally — see `metrics.py` docstring), since the real challenge test set presumably has proper prescription data. Not re-checked against later epochs (6-9) individually since val loss was already worse there and time did not allow it.

**Critical: the deadline is 2026-08-15 — this was discovered mid-session to be 1 day away, not the 3-5 days assumed in earlier sessions.** Steps 1-10 of the checklist are now done. Everything remaining (Steps 11-12) requires the user's own Grand Challenge account and cannot be done by the assistant directly.

### Files created / changed

| Path | Change |
|---|---|
| `src/training/losses.py` | Added `OutsideBodyL1Loss`, wired into `DoseLoss` as a third term. |
| `configs/task1_photon_ct.yaml` | Added `loss.w_outside: 0.2`. |
| `checkpoints/task1_photon_ct/best_pre_outside_body_fix_epoch4.pt`, `last_pre_outside_body_fix_epoch9.pt` | **New.** Backups of the pre-fix checkpoints, kept as a rollback fallback. |
| `checkpoints/task1_photon_ct/best.pt`, `last.pt` | **Overwritten** by the fine-tune run — `best.pt` is now epoch 5 (post-fix). |
| `logs/train_task1_photon_ct_outside_body_fix.log` | **New.** Full fine-tune training log. |
| `logs/evaluate_epoch5_fix.log` | **New.** Full metric suite against the epoch-5 post-fix checkpoint. |
| `docker/model/task1_photon_ct/last.pt` | **Replaced** with the epoch-5 post-fix checkpoint. |
| `docker/doserad2026_task1_photon_ct_2026-08-14T10-18-36.670007874-04-00.tar.gz`, `docker/model.tar.gz` | **New.** Final upload-ready packages. |

### Open questions / what's next

- Steps 11 (Preliminary Testing Phase submission) and 12 (Final submission) require the user's own Grand Challenge account: create/verify account → join DoseRAD2026 → create Algorithm entry (GPU: NVIDIA A10G 24GB) → upload the image tarball + `model.tar.gz` as a separate Model → wait for "Active" → submit. This is the critical path with the least buffer given the 2026-08-15 deadline.
- Real per-beam timing on actual AWS g5 hardware still unconfirmed (only measured on this server's A100 and via local Docker, ~4.7s for 4 CPs = ~1.2s/CP — worth watching closely on the real Preliminary Testing Phase run since it's close to the 1s/beam limit, unlike earlier looser estimates).
- Did not re-check whether epoch 6 (val loss 0.0969, nearly tied with epoch 5's 0.0968) might have a better gamma_index_pass_rate than epoch 5 — time did not allow it; epoch 5 was used based on val loss ranking alone.

---

## Session: 2026-08-10 — Training run finished; full metric evaluation; real checkpoint swapped into Docker; packaging scripts written

### Steps taken, in order

1. Checked live state (not the checklist, which was stale): the relaunched training run from the 2026-08-06 session had actually finished all 10 epochs cleanly by 2026-08-07 16:33, with no further crashes (147/36180 steps skipped in the final epoch, consistent with the earlier NaN-guard design). `CHECKLIST.md`/`PROGRESS_LOG.md` hadn't been updated to reflect this.
2. Ran `src/evaluation/evaluate.py --checkpoint checkpoints/task1_photon_ct/best.pt` for the first time -- the full official metric suite (Masked MAE, IDD curve distance, Stratified Plan MAE, Gamma Index 1%/1mm) had never actually been run before now; training only ever logged its own loss components (masked_mae/body_l1), not the standalone metrics.
3. Copied `best.pt` (epoch 4, the lowest validation loss of the run) into `docker/model/task1_photon_ct/last.pt`, replacing the stale Aug 6 placeholder checkpoint used only for plumbing validation.
4. Rebuilt the Docker image and re-ran the same real-patient end-to-end test used in the previous session, this time against the real trained weights, confirming the container correctly loads and serves `best.pt` ("Loaded checkpoint /opt/ml/model/task1_photon_ct/last.pt (epoch 4)" in the logs) and still produces correctly-shaped output.
5. Wrote `docker/do_build.sh`, `docker/do_test_run.sh`, `docker/do_save.sh` -- a repeatable one-command packaging workflow matching the official reference repo's own tooling (adapted: no `jq` dependency since we don't need their multi-run `predictions.json` format, our own `evaluate.py` already covers metrics). Added `docker/test-data/input/` as a permanent, committed test fixture (real CT + 4 real control points + 9 placeholder slots, copied from this session's scratchpad) so `do_test_run.sh` is self-contained and repeatable, not dependent on manually reconstructing a test case each time.
6. Hit and fixed one real bug in `do_test_run.sh` while validating it: the freshly-created host-owned `test-data/output/` directory wasn't writable by the container's differently-privileged internal user, causing `PermissionError: [Errno 13] Permission denied: '/output/images'` on the first real `/invoke` call. Fixed with `chmod -R 777` on the output directory before starting the container (matches the same class of permission issue hit and fixed for the model directory in the previous session).
7. Re-ran `do_test_run.sh` after the fix -- full clean pass: build → health check (200 on 3rd attempt) → invoke (**1.20s for 4 control points, ~0.30s/CP** including full HTTP/curl overhead, not just the model) → output written and verified.

### Findings -- accuracy is a real concern, not yet submission-quality

Full validation-set metrics (8 held-out patients, all 540 control points each, `best.pt`/epoch 4):

| Metric | Value | Read |
|---|---|---|
| `masked_mae` | **0.0953** (9.5%) | Level 1, per-CP, high-dose region only. Matches training's own logged val loss for this epoch. Workable but not tight -- competitive dose-prediction models typically target low single digits. |
| `idd_curve_distance` | **126.5** | Expected to be roughly O(0.01-1) (RMSD normalized by peak GT IDD). A value this large is a strong signal of gross over-prediction somewhere along the beam depth path, not a rounding-level issue. |
| `stratified_plan_mae` | **0.0716** (7.2%) | Level 2, full-plan (sum of a patient's 540 CPs). Roughly in the same ballpark as masked_mae. |
| `gamma_index_pass_rate` | **0.0135** (1.3%) | Level 2, strict 1%/1mm local-normalization criterion. Even accounting for how strict 1%/1mm is, a ~1% pass rate is a near-total fail, not just "room to improve." |

This lines up with a finding surfaced while building the demo artifact earlier (2026-08-06): at that time, ~95.8% of predicted dose *mass* sat outside the body mask, visible as a checkerboard/moire artifact in open air. That check was only done on one control point at epoch 0; **this session's full evaluation shows the pattern is still driving real metric damage at epoch 4** (the best checkpoint from the completed run) -- the `idd_curve_distance` blowup (evaluated along the whole beam depth path, not body-masked) and the collapsed gamma pass rate (which compounds per-CP errors across all 540 summed control points in a full plan) are both consistent with the model still spreading meaningful dose mass into air/outside the body, not just at the tissue boundary.

**Likely root cause, not yet confirmed**: `src/training/losses.py`'s masked MAE loss only rewards accuracy inside the high-dose region -- there is no explicit penalty for predicting dose in the background/outside-body region, so the model has no gradient signal telling it to suppress that mass. This was flagged as a hypothesis in the 2026-08-06 session and is now looking like the actual explanation, not just a minor early-epoch artifact that would resolve with more training (validation loss had already plateaued by epoch 4, and later epochs 5-9 got *worse* on val despite train loss continuing to fall -- classic overfitting past that point, not underfitting).

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/checkpoints/task1_photon_ct/best.pt`, `last.pt` | Pre-existing (written by the training run that finished 2026-08-07, not this session) -- first time actually read/evaluated this session. |
| `DoseRAD/logs/evaluate_best.log` | **New.** Full-suite evaluation output against `best.pt`. |
| `DoseRAD/docker/model/task1_photon_ct/last.pt` | **Replaced.** Was the Aug 6 placeholder (epoch 0/1); now `best.pt` (epoch 4). |
| `DoseRAD/docker/do_build.sh` | **New.** One-command image build. |
| `DoseRAD/docker/do_test_run.sh` | **New.** One-command build + local end-to-end invoke test against `docker/test-data/`. |
| `DoseRAD/docker/do_save.sh` | **New.** Packages the image as a gzipped tarball and the model directory as `model.tar.gz`, matching Grand Challenge's separate-container-and-model upload convention. |
| `DoseRAD/docker/test-data/input/` | **New (committed fixture).** Real CT + 4 real control points + 9 placeholder slots, moved out of the session scratchpad so testing is self-contained and repeatable. |

### Checklist status after this session

- [x] **Step 8 — First training run**: training itself is complete (10/10 epochs, no crashes). Marking done in the sense of "a full run completed" -- but see Findings: the resulting model is not yet submission-quality on the fuller Level 2 metrics, so this isn't "accuracy validated as good," just "the run finished."
- [~] **Step 10 — Docker submission container**: real trained checkpoint now baked in, one-command build/test/save workflow in place. Mechanically complete; blocked on the same accuracy concern as Step 8 before this should actually be submitted.

### Open questions / next steps

- **Decision needed**: submit the current model now (guarantees a valid submission before the 2026-08-15 deadline, meets the speed requirement comfortably, but accuracy -- especially the near-zero gamma pass rate -- is weak), or spend some of the remaining ~5 days trying to fix the out-of-body over-prediction (most likely fix: add an explicit penalty term for predicted dose outside `body_mask` to `src/training/losses.py`, then retrain) at the risk of not finishing in time. Not decided yet -- surfaced to the user, awaiting direction.
- If retraining is pursued: start from the existing checkpoint rather than from scratch (5 days doesn't comfortably fit a fresh 10-epoch run at ~2.5hr/epoch -- that alone is ~25hrs, feasible but tight alongside re-evaluation and re-testing).
- Real per-beam timing still only confirmed on this server's A100, not actual AWS g5 hardware (Step 11).
- Step 11/12 need the user's own Grand Challenge account.

---

## Session: 2026-08-06 (continued) — Phase C, Step 10: Docker submission container built and validated end-to-end

### Steps taken, in order

1. User provided the real submission instructions page and pulled the official reference implementation from `https://github.com/DoseRAD2026/example-submission`. This resolved the earlier-flagged blocker: the exact input/output contract is an HTTP `invoke` API (long-lived server, not a one-shot CLI), reading up to 10 source images plus one beam-level metadata JSON from `/input`, writing up to 10 stacked dose-map `.mha` files to `/output`. For photon, timing is per control point (matches this codebase's existing granularity everywhere else).
2. Built `docker/app.py` (FastAPI server, `/health` + `/invoke`, loads the model once at startup) and `docker/process.py` (inference logic), reusing the *exact* training-time `src/data/transforms.py` / `src/data/beam_encoder.py` / `src/models/model_factory.py` components rather than reimplementing preprocessing -- the single most important correctness property, since any drift between train-time and inference-time preprocessing would silently invalidate predictions.
3. Wrote `docker/Dockerfile` (build context = project root, so `src/` and `configs/` can be copied in) and `docker/requirements.txt`. Added `.dockerignore` excluding `data/`, `checkpoints/`, `.venv/`, and critically `Hugging Face Token.txt` (must never enter an image layer).
4. Built the image locally (Docker confirmed available) and constructed a realistic local test case from real patient `1ABB006` (real CT in input slot 1, 1x1x1 placeholders in slots 2-10, 4 real control points from beam 0 with `output_info` added) to validate end-to-end before trusting the format.
5. **Found and fixed 4 real bugs during this local test**, in order:
   - **`CenterCropOrPad` assumed a `dose` key always exists** in the sample dict -- true during training, false at inference (no ground truth to crop). `KeyError: 'dose'` on first real invoke. Fixed by only touching `dose` when present -- correct general fix since the transform is shared by both paths, not a workaround.
   - **Model directory permissions**: `docker/model/` was created with the host user's default restrictive permissions (no `o+r`), but the container runs as a different, unprivileged user, so it silently found zero checkpoint files. Fixed with `chmod -R a+rX` -- the official reference test script does the exact same defensive chmod before running, for the same reason.
   - **Cold-start latency risk**: first real `/invoke` after container startup took ~2.4-3.4s for 4 control points (vs. ~0.1s/CP steady-state) because CUDA/cuDNN kernel selection for both the model *and* `PhotonBeamEncoder.encode()` was paid on the first real call instead of at startup. If a real evaluation invokes with as few as one beam per call, that cold-start cost alone could exceed the 1s/beam limit on the very first scored beam. Fixed by adding a proper warmup pass in `init_model()` that exercises the *exact* same code path (encoder + model) with dummy data before the server reports healthy.
   - **The real per-beam bottleneck wasn't the model at all**: mapping a prediction from our internal 256^3 grid back onto each patient's native CT grid (required: "output dose grid must match input image exactly") was originally done via a general `sitk.Resample()` call per control point -- CPU-bound, ~0.37s/CP end-to-end vs ~0.1s/CP for model+encoder alone, eating most of the safety margin under the 1s limit. Since every real patient checked so far has spacing that already matches our training grid (no actual resampling ever needed, only crop/pad), added `CenterCropOrPad.restore()` -- a pure inverse of the existing forward crop/pad, done as a GPU tensor op with no CPU round-trip -- as a fast path, falling back to the general `sitk.Resample` path only if spacing genuinely differs (kept for robustness against an unknown hidden-test-set patient). **Round-trip-tested `restore()` against real patient shapes before trusting it**: caught a subtle bug where Python's floor division doesn't negate cleanly for odd size differences (`7//2=3` but `-(-7//2)=4`), which would have silently misaligned the output by a voxel on any axis with an odd crop/pad difference. Fixed by mirroring the forward pass's exact per-branch arithmetic instead of deriving the inverse from one signed offset.
6. **Profiled further after the fast-path fix showed only a partial improvement** (0.33s/CP, expected much better): found two more real, avoidable costs -- `ndarray.astype(np.float32)` on an already-float32 array was silently doing a full extra copy (numpy's `astype` copies by default even when the dtype already matches, ~40ms/CP wasted), and `run()` was doing a *second* full `sitk.GetArrayFromImage`/`GetImageFromArray` round trip just to apply the `minimum_cutoff` threshold, duplicating work already available (~80ms/CP wasted). Fixed by applying the cutoff as a single `torch.where` on the GPU tensor before the one necessary CPU transfer, and dropping the redundant `.astype()`.
7. Rebuilt and re-ran the same local test after each fix to confirm both correctness (output geometry/values unchanged) and the actual measured speedup, rather than assuming either.

### Findings

- **Final measured per-control-point time inside the actual container: ~0.16s/CP** (down from ~0.33s/CP before the last two fixes, ~0.85s/CP before the warmup fix) -- roughly 6x under the 1s/beam limit on this server's A100, with the model+encoder alone accounting for ~85ms of that and the CT-to-original-grid restore for the rest. Even a pessimistic slowdown estimate for AWS g5's weaker A10G leaves real margin, though this should still be confirmed on the actual target hardware during Step 11.
- End-to-end output format validated directly: all 10 output slots present, the used slot is a proper 4D stack via `sitk.JoinSeries` with size `(249,246,246,4)` exactly matching the input CT's native grid (spacing, origin all match), unused slots are 1x1x1 placeholders matching the official reference's own convention.
- The checkpoint currently baked into `docker/model/task1_photon_ct/last.pt` is the actively-training run's early checkpoint (epoch 0/1, not converged) -- used only to validate plumbing/format/speed. **Must be swapped for a real, trained checkpoint before actual submission.**
- `docker/do_build.sh` / `do_test_run.sh` / `do_save.sh` (packaging helper scripts matching the official reference's workflow) not yet written -- the local test above was done with manual `docker build`/`docker run`/`curl` commands instead, sufficient to validate correctness but not yet a repeatable one-command workflow.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/docker/app.py` | **New.** FastAPI inference server (`/health`, `/invoke`), loads model once at startup. |
| `DoseRAD/docker/process.py` | **New.** Inference logic: reads the batched input contract, runs one model call per control point, restores predictions to each patient's native grid, writes the stacked output contract. |
| `DoseRAD/docker/Dockerfile` | **New.** Built from project root context so `src/`/`configs/` can be copied in; required `org.grand-challenge.api-method="invoke"` label. |
| `DoseRAD/docker/requirements.txt` | **New.** `fastapi`, `uvicorn`, `SimpleITK`, `monai`, `pyyaml`, `numpy` (torch deliberately omitted -- already in the pytorch base image, avoiding a redundant/risky reinstall). |
| `DoseRAD/.dockerignore` | **New.** Excludes `data/`, `checkpoints/`, `.venv/`, `Hugging Face Token.txt`, etc. from the build context. |
| `DoseRAD/docker/model/task1_photon_ct/last.pt` | **New (placeholder).** Current in-training checkpoint, for plumbing validation only -- not the final submission checkpoint. |
| `DoseRAD/src/data/transforms.py` | `CenterCropOrPad.__call__` now only touches `sample["dose"]` when present (bug fix). Added `CenterCropOrPad.restore()`, the inverse crop/pad used by the container's fast path, round-trip tested against real patient shapes. |
| `DoseRAD/docker/process.py` | Cutoff thresholding moved onto the GPU before the CPU transfer; removed a redundant `.astype()` copy (both speed fixes, see Findings). |

### Checklist status after this session

- [~] **Step 10 — Docker submission container**: core plumbing built and validated locally end-to-end (real patient input in, correctly-formatted output out, well within the speed limit). Not yet complete -- needs the real trained checkpoint swapped in, and the `do_build.sh`/`do_test_run.sh`/`do_save.sh` packaging scripts written for a repeatable workflow.

### Open questions / next steps

- Swap in a real trained checkpoint once Step 8's training run reaches a good validation accuracy (currently mid-epoch-1, healthy, no NaN).
- Write `docker/do_build.sh` / `do_test_run.sh` / `do_save.sh` so packaging for upload is a repeatable one-command workflow, matching the official reference's own tooling.
- Confirm real per-beam timing on an actual AWS g5 instance during Step 11 (Preliminary test phase) -- the A100 numbers here are a strong signal, not a substitute for testing on the real target hardware.
- Step 11 (join the challenge, upload the container + model tarball, submit) needs the user's Grand Challenge account -- not something that can be done from here.

---

## Session: 2026-08-06 — Phase B, Step 8: First run diverged to NaN; root-caused, fixed, relaunched

**Context:** picking back up after the 2026-07-30 pause. 9 days left until the 2026-08-15 deadline.

### Steps taken, in order

1. **Checked on the background run from 2026-07-30 (PID 2664245) and found it had already finished — badly.** The process was no longer running; `checkpoints/task1_photon_ct/last.pt` was last written 2026-08-02 10:45. The log showed `masked_mae=nan body_l1=nan total=nan` for every logged step of every one of the 15 epochs' `[train]`/`[val]` summary lines.
2. **Traced the exact divergence point**: `grep`'d the ~27k-line log for the first `nan` — found at **epoch 0, step 29980/36180** (83% through the very first epoch). The step immediately before it was healthy: `masked_mae=46.76`. Every one of the ~69 hours of training after that point was pure wasted GPU time on an already-broken model — a full week effectively lost.
3. **Root-caused before touching any code**, per usual practice: read `losses.py`, `trainer.py`, `unet3d.py`/`model_factory.py`, `hf_dataset.py`, `transforms.py`. Found:
   - `MaskedMAELoss`/`BodyMaskedL1Loss` normalize each sample's L1 error by that sample's `dose.max()` ("beam_max") — correct concept (matches the official metric), but nothing in `transforms.py` ever rescales `dose` itself, so `beam_max` stays in raw Gy units.
   - `NormalizeCT` puts CT into `[0, 1]`, but there is no equivalent for dose.
   - `build_unet3d()` uses MONAI's `UNet` with **no final activation**, so a freshly-initialized model's output is unconstrained, typically order 0.1-1.
   - **Measured real per-CP dose scale directly from 120 real dose files across 8 patients**: max value per control point is tightly clustered at **5.4e-5 to 1.3e-4 Gy** (min-to-max spread only ~2.3x — not wild per-sample outliers, a consistent global scale issue). So the model's raw output was ~10,000-100,000x larger than what it was being asked to match, and the per-sample-normalized loss (dividing by that tiny `beam_max`) amplified the resulting error into the huge, unstable values seen in the log (`total≈50` at the point of divergence — a well-scaled normalized loss should be O(1) or smaller). Under `torch.autocast` fp16, this eventually overflowed on one step, and — critically — there was **no NaN/Inf guard anywhere in the optimizer step**, so that one bad step's corrupted gradient got applied via `optimizer.step()` and permanently poisoned every model weight (matches the log exactly: no recovery, ever, for the remaining 14.17 epochs).
4. **Reported this full chain of evidence before fixing anything** (per established practice) — this section *is* that report, written contemporaneously.
5. **Implemented three fixes**:
   - **`src/data/transforms.py`**: new `ScaleDose(scale=10000.0)` transform, multiplying `dose` by a fixed constant right after load so targets sit near O(1). All the official metrics (Masked MAE, IDD curve distance, Gamma Index, Stratified Plan MAE) are relative/normalized, so this is invisible to them as long as pred/target are compared in consistent units — confirmed by inspecting each metric in `metrics.py`. **Any future inference/submission code must divide the raw model output by this same factor to recover real Gy** — flagged in the transform's docstring and the config comment so Step 9/10 doesn't miss it.
   - **`src/training/trainer.py`**: `train_epoch()` now checks `torch.isfinite(loss)` before every `backward()` call, and checks `torch.isfinite(total_norm)` (the return value of `clip_grad_norm_`) before every `optimizer.step()` — either check failing skips that step entirely (zeroed grads, logged, counted in a new `self.skipped_steps`) instead of ever applying a corrupting update. This is a permanent safety net regardless of what causes a future bad batch. Also added `save_every_steps` (default 2000) so a mid-epoch problem no longer risks losing an entire multi-hour epoch's progress — previously `last.pt` only saved once per epoch.
   - **`configs/task1_photon_ct.yaml`**: added `dose_scale: 10000.0`, added `save_every_steps: 2000`, reduced `epochs: 15 -> 10` (given a week was just lost, a shorter, provably-stable run before committing more days felt safer than repeating the same size of bet).
   - Wired `ScaleDose` into both `scripts/train.py`'s and `src/evaluation/evaluate.py`'s transform pipelines so train/val/eval all stay consistent.
6. **Validated the fix empirically before relaunching** (300 real steps, real data, same config): loss dropped from `total=5.03` at init to a stable `total≈0.12-0.13` by step 300 — monotonic, no instability, `0` non-finite steps out of 300. Compare to the old run's `total≈50+` at the same relative point in training. Strong, direct evidence the scale mismatch was in fact the root cause, not just a plausible theory.
7. **Archived (not deleted) the broken run's artifacts**: `checkpoints/task1_photon_ct/last.pt` -> `last_BROKEN_nan_run_2026-08-02.pt`, `logs/train_task1_photon_ct.log` -> `logs/train_task1_photon_ct_BROKEN_nan_run_2026-08-02.log`.
8. **Relaunched training** the same way as before (`nohup ... & disown`), confirmed running: **PID 3843369**, GPU 0 at 80% utilization, 14.7GB VRAM. Watched the first ~160 real steps live: the NaN-guard immediately proved itself, catching and cleanly skipping 3 non-finite grad-norm steps right at the very start (expected — this is GradScaler's normal calibration transient), then loss fell smoothly (`masked_mae` 2.29 -> 0.46 by step 160) with zero further skips.

### Findings

- **The entire 2026-07-30 training run (~69 hours of GPU time) was wasted** — it diverged to NaN 83% through epoch 0 and every subsequent epoch trained on an already-corrupted model. No usable checkpoint survived it.
- Root cause was a **target-scale mismatch** (raw ~1e-5 Gy dose vs. an unconstrained ~O(1) model output) combined with **no NaN/Inf guard in the training loop** — either alone might have been survivable; together they turned one bad batch into total, silent, unrecoverable model corruption that ran undetected for days because checkpoints only saved once per epoch.
- Real per-CP dose max is consistently ~5e-5 to 1.3e-4 Gy across patients (measured directly from files, not assumed) — useful reference number for any future scale-related debugging.
- 3 of 4 server GPUs remain idle during this run — still available for Step 9 profiling work in parallel.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/data/transforms.py` | New `ScaleDose` transform (root-cause fix). |
| `DoseRAD/src/training/trainer.py` | NaN/Inf-safe optimizer step (loss + grad-norm finiteness checks, skip-and-log instead of corrupt); `save_every_steps` mid-epoch checkpointing; `skipped_steps` counter. |
| `DoseRAD/configs/task1_photon_ct.yaml` | Added `dose_scale: 10000.0`, `save_every_steps: 2000`; `epochs: 15 -> 10`. |
| `DoseRAD/scripts/train.py` | Wired `ScaleDose` into the transform pipeline; passes `save_every_steps` to `Trainer`. |
| `DoseRAD/src/evaluation/evaluate.py` | Wired `ScaleDose` into `build_val_dataset`'s transform pipeline for consistency with training. |
| `DoseRAD/checkpoints/task1_photon_ct/last.pt` | Archived as `last_BROKEN_nan_run_2026-08-02.pt` (all-NaN weights, kept as evidence, not used). |
| `DoseRAD/logs/train_task1_photon_ct.log` | Archived as `..._BROKEN_nan_run_2026-08-02.log`; fresh log started for the relaunch. |
| `DoseRAD/CHECKLIST.md` | Marked Steps 1-4 `[x]` (were done in an earlier session but never checked off). |

### Checklist status after this session

- [~] **Step 8 — First training run**: relaunched (PID 3843369) after fixing the NaN divergence; 10 epochs targeted (~28hr at the previously-measured ~2.8hr/epoch, to be reconfirmed once a full epoch completes under the new code path). Not yet complete.

### Open questions / next steps

- Check back on this run within the next few hours (not blind for days this time) to confirm stability holds well past the ~30k-step mark where the previous run broke.
- Once epoch 0 completes, spot-check with `src/evaluation/evaluate.py --checkpoint checkpoints/task1_photon_ct/last.pt --cp_stride 20 --max_patients 2`.
- Started **Step 9 (speed optimization)** groundwork in parallel, on GPU 1 (idle, avoids contending with training on GPU 0) — see below.
- Given 9 days left and this week's loss, keep the remaining schedule tight: reserve realistic time for Step 9 (speed), Step 10 (Docker), and Step 11/12 (submission) rather than over-investing further in training epochs beyond what's needed to clear a reasonable accuracy bar.

### Step 9 groundwork (same session)

- Wrote `scripts/inference.py`: single-control-point inference (CT + beam params -> dose in real Gy, dividing by `dose_scale`). Confirmed from `DoseRAD2026_Complete_Reference.md`'s "Data geometry notes" that for photon/VMAT, the thing timed against the 1s/beam limit is **one control point**, not a full 180-CP arc -- matches the existing per-sample granularity everywhere else in the codebase.
- **Benchmarked real forward-pass latency on the A100**: `0.099s/call` (30-call benchmark, after warmup, includes beam encoding + model forward) -- about **10x under the 1-second limit**. Used the archived broken checkpoint for this since weight values don't affect compute time, only architecture/shapes do.
- AWS g5 uses an A10G GPU, weaker than this server's A100 (roughly 2.5-4x slower for typical fp16 workloads) -- even a pessimistic slowdown estimate leaves comfortable headroom under 1s, but this is an estimate, not a measurement on the real target hardware. Should be confirmed for real during Step 11's preliminary submission (g5 is selectable in Grand Challenge submission settings).
- **Blocker found, needs the user**: the exact Docker/algorithm I/O contract (input file naming/socket conventions, output format Grand Challenge expects from `process.py`) is not in `DoseRAD2026_Complete_Reference.md` -- the doc says "submission instructions are locked -- must join the challenge first". Can't finish `docker/process.py` correctly without this; the user needs to log into the Grand Challenge platform (their account, not accessible to me) and pull the algorithm template/interface spec so Step 10 can be built against the real contract instead of a guess.

---

## Session: 2026-07-30 (continued) — Phase B, Step 8: First real training run

### Steps taken, in order

1. **Measured real per-step throughput before committing to any epoch count**, using the actual full 36,180-sample train split (67 patients) rather than trusting the earlier tiny smoke-test timings: `0.309s/step` at `num_workers=8`. Checked system resources first -- 503GB RAM (421GB already in page cache, enough to hold most of the 384GB dataset after first read), 96 CPU cores, A100-40GB GPU -- so worth checking whether more DataLoader workers would help.
2. Tried `num_workers=16` and `32`: `0.278s/step` and `0.281s/step` -- plateaus around 16, confirming the pipeline is now **compute-bound, not I/O-bound** (more workers wouldn't help further). Extrapolated: **~2.8 hours per full epoch**.
3. **Found and fixed a real path bug while setting up the timing test**: `configs/task1_photon_ct.yaml`'s `data_root: data/photon_ct` is relative to the current working directory, but the actual downloaded data lives at `/home/jmasina/DoseRAD/data/photon_ct` -- one level **above** this project directory (`DoseRAD/DoseRAD`), not inside it (a leftover of how `scripts/download_data.py`'s default output path was set up in the Step 1 session). Running `python scripts/train.py` from the natural project root would have failed immediately with `FileNotFoundError`. Fixed by making `data_root` an absolute path in the config, since this is a single-server deployment where portability isn't a current concern.
4. Updated `configs/task1_photon_ct.yaml`: `num_workers: 4 -> 16` (measured optimum), `epochs: 50 -> 15`. **50 was a placeholder from Step 6** written before real timing existed; at 2.8hr/epoch, 50 epochs = ~140 hours (~5.8 days) -- too much of the remaining ~16-day budget to commit to blind on a "first" run per `CHECKLIST.md`'s own framing (Step 8 says "iterate ... before optimizing for speed", implying this run's job is to validate the pipeline and get a first accuracy read, not be the final model). 15 epochs = ~42 hours (~1.75 days), and `scripts/train.py --resume` already supports continuing further once this run's results are in.
5. **Dry-ran the actual `scripts/train.py` CLI end-to-end** (not just the individual component smoke tests from Steps 5-7) for 90 seconds before launching the real multi-day run, specifically to catch any CLI-level bug (argument parsing, config loading, checkpoint dir setup) that the component-level smoke tests wouldn't exercise. Caught nothing new (the path bug above was found before this step, while building the timing script), but confirmed: dataset loads (40,500 CPs / 75 patients), the 67/8 train/val split matches the earlier evaluate.py smoke test's split (same seed, same shared `split_patient_ids()` helper -- consistency confirmed), and training steps run cleanly.
6. **The dry run's loss curve was a strong signal the whole pipeline (including the just-fixed beam encoder) is working correctly together**: `masked_mae` dropped from 42,927 -> 3,728 over the first 300 steps, monotonically, no NaNs/instability. This is the running average since step 1, so the sharp early drop is expected (model rapidly learns to output near the tiny target scale rather than random large values) -- but a monotonic, non-erratic drop this early is exactly what you'd want to see before committing to a long unattended run.
7. Launched the real run in the background (`nohup python -u scripts/train.py --config configs/task1_photon_ct.yaml > logs/train_task1_photon_ct.log 2>&1 & disown`), decoupled from this session the same way the Step 1 dataset download was. Confirmed running: PID 2664245, GPU utilization 63%, 11.6GB VRAM, loss curve continuing the same healthy downward trend seen in the dry run.

### Findings

- **Real epoch time: ~2.8 hours** (67 train patients x 540 control points = 36,180 samples/epoch, batch size 1, AMP on, `num_workers=16`, A100-40GB). This number should drive all further training-time budgeting for the rest of the project.
- Two real bugs caught in this session before they could waste a multi-hour background run: the `data_root` relative-path issue (step 3 above) and the placeholder 50-epoch config value that hadn't been reconciled with real timing (step 4). Both are the kind of thing that would have been very costly to discover only after checking back on a run that had already run for hours or days against a subtly wrong setup.
- 3 of the server's 4 GPUs are currently idle during this run (only 1 used). Worth keeping in mind for later: either running future experiments/hyperparameter variants in parallel on the other GPUs, or (bigger lift) multi-GPU data-parallel training if more speed is needed -- not pursued now since single-GPU epoch time is already within a workable budget.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/configs/task1_photon_ct.yaml` | `data_root` changed to an absolute path (bug fix); `num_workers: 4 -> 16`; `epochs: 50 -> 15` (both based on measured real timing, not placeholders). |
| `DoseRAD/logs/train_task1_photon_ct.log` | **New.** Live training log for the background run (`nohup` stdout+stderr). |
| `DoseRAD/checkpoints/task1_photon_ct/` | Will contain `last.pt` (every epoch) and `best.pt` (lowest val loss so far) once the first epoch completes (~2.8hr from launch). |

### Checklist status after this session

- [~] **Step 8 — First training run**: launched, in progress (background, ~42hr estimated for the initial 15 epochs). Not yet checked off complete -- needs the run to actually finish and `evaluate.py` to be run against a real trained checkpoint before calling this step done.

### Open questions / next steps

- **Check back on training progress** (`tail logs/train_task1_photon_ct.log`, or `nvidia-smi` to confirm it's still alive) periodically rather than waiting the full ~42 hours blind. Once at least one epoch's checkpoint exists, worth running `src/evaluation/evaluate.py --checkpoint checkpoints/task1_photon_ct/last.pt --cp_stride 20 --max_patients 2` as a quick spot-check of real (not just training-loss) accuracy trend, without waiting for the full run to finish.
- Since this runs unattended in the background, the ~1.75 days it takes can overlap with starting **Step 9 (speed optimization)** groundwork and **Step 10 (Docker packaging)** structure -- neither strictly needs the fully-trained model to begin, just the model architecture, which already exists.
- After this first run completes: decide whether to extend via `--resume` (more epochs), adjust hyperparameters/loss weights based on the val metrics, or move on -- per `CHECKLIST.md`'s Step 8 guidance to iterate before shifting focus to speed.

---

## Session: 2026-07-30 (continued) — Bug fix: MLC left/right aperture sign convention

### Context

While explaining `1ABB006.json`'s contents to the user in detail (a walkthrough, not a bug hunt), pulled real `mlc_left_int_mm`/`mlc_right_int_mm` values to show as examples and noticed something inconsistent with `beam_encoder.py`'s assumption: both fields can independently be negative or positive (e.g. one real leaf pair had `left=-51, right=-33` -- a valid, open, off-center aperture). The encoder's `encode()` computed the aperture as `[-left, right]` (negating `left`, on the assumption it was a distance-from-center magnitude), which is only correct if `left` is always non-negative.

### Steps taken, in order

1. Quantified the scope before touching any code: computed, across all 180 control points of `1ABB006` beam 0, what fraction of leaf-pair/control-point entries produce an inverted (negative-width, therefore spuriously "closed") range under the existing `[-left, right]` formula vs. a direct `[left, right]` interpretation. **8.9% inverted under the old formula vs. 0.02% under the direct interpretation** -- strong evidence `mlc_left`/`mlc_right` are raw signed leaf-tip coordinates, not magnitudes, and the old formula was wrong.
2. Reported this finding to the user with the numbers before changing anything, per the project's standing practice of not silently fixing things. User confirmed: go ahead, and be thorough about it so this doesn't need revisiting later.
3. Fixed `PhotonBeamEncoder.encode()` in `src/data/beam_encoder.py`: aperture is now `(lateral_iso >= left_at_voxel) & (lateral_iso <= right_at_voxel)`, i.e. `[left, right]` directly, no negation.
4. **Re-validated against real dose data**, same dose-overlap methodology as the original Step 4 calibration (`dose[mask].sum() / dose.sum()`), across 204 wide-aperture control points spanning all 6 beams of both downloaded sample patients (1ABB006, 1ABB011):
   - **New formula won outright on 204/204 control points** (0 for the old formula, 0 ties).
   - **Mean dose-overlap: 3.6% (old, buggy formula) -> 77.3% (fixed formula)** -- the old aperture mask was capturing almost none of the real dose; the fixed one captures the large majority of it, which is what a correct MLC-shaped aperture should do.
5. **Re-checked the `rotation_sign=-1.0` gantry-rotation calibration** against the *fixed* aperture formula (since the original calibration in Step 4 was run against the buggy version, and the two conventions are logically independent, but worth confirming neither was compensating for the other). Result: **rotation_sign=-1.0 still wins, 195/204**, an even stronger margin than the original Step 4 result (553/600 = 92.2% -> 195/204 = 95.6%). Confirms the rotation-sign finding was real and not an artifact of the aperture bug.
6. Updated `PhotonBeamEncoder`'s docstring to document the corrected convention and both sets of real numbers (old vs. new formula overlap; re-checked rotation-sign win rate) directly in the code, matching how the original calibration was documented.
7. **Re-ran every downstream smoke test** that depends on the beam encoder, to make sure nothing silently relied on the old (wrong) mask shape:
   - Beam-encoder GPU timing test (Step 5): still ~0.008s/call on GPU, 100% CPU/GPU agreement -- fix didn't affect performance.
   - Full training-loop smoke test (Step 6): forward/backward/optimizer/checkpoint save+reload all still run cleanly on the corrected masks.
   - Full evaluation-pipeline smoke test (Step 7): all metric functions and `evaluate()` still run cleanly on the corrected masks.

### Findings

- This was a real, significant, previously-undetected correctness bug affecting the beam-conditioning input the model was about to be trained on -- not a style issue. Left uncaught, Step 8's training run would have quietly learned from a systematically wrong aperture signal for a large fraction of control points, capping achievable accuracy in a way that would have been very hard to diagnose after the fact (the model would have "worked," just underperformed, with no obvious error to point to).
- The original Step 4 rotation-sign calibration's conclusion held up under re-checking, which is reassuring -- it means that finding was genuine and not coincidentally compensating for this separate bug.
- General lesson for the rest of this project: when a data field's sign/unit convention isn't documented anywhere (as was true here -- the reference doc and JSON give no spec for `mlc_left_int_mm`/`mlc_right_int_mm`'s sign convention), treat any assumption about it as unverified until checked against real values, not just "does the code run without crashing."

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/data/beam_encoder.py` | **Bug fix.** `encode()`'s aperture formula changed from `[-left, right]` to `[left, right]`. Docstring updated with the investigation and re-validation numbers. |

### Checklist status after this session

- Step 4 (Photon beam encoder) status unchanged (already marked complete) but its underlying correctness is now substantially stronger -- this was a fix to already-"done" work, not a new step.

### Open questions / next steps

- Proceed to **Step 8 — first real training run** now that the beam-conditioning input is verified correct, not just functional.
- No other unverified sign/unit conventions are currently known in the pipeline, but worth staying alert for similar issues (e.g. if proton-task work starts later, `ProtonBeamEncoder` is still an untouched stub with its own unverified assumptions to make from scratch).

---

## Session: 2026-07-30 (continued) — Phase B, Step 7: Evaluation metrics

### Steps taken, in order

1. Re-read the exact metric definitions in `DoseRAD2026_Complete_Reference.md` (Level 1: Masked MAE, IDD Curve Distance; Level 2: Stratified Plan MAE, 3D Local Gamma Index 1%/1mm, DVH-Based Clinical Score).
2. **Checked whether the data these Level 2 metrics need actually exists locally** before writing code for them: searched every one of the 77 downloaded patients' folders for structure/segmentation/prescription files (`*struct*`, `*rtss*`, `*prescri*`) and inspected the beam JSON's full key structure. **Confirmed: none exists.** Each patient folder only has `image/ct.mha`, `image/mr.mha`, `dose/Dose_B{beam}_CP{cp}.mha`, and `<PID>.json` (beam geometry only — no prescription dose field, no contours). This is a real, load-bearing gap, not an oversight to code around silently.
3. Refactored `src/data/beam_encoder.py`: extracted the source-position/axis-vector setup shared by `encode()` into a new `_beam_geometry()` helper, and exposed a public `beam_depth()` method (per-voxel depth in mm along the central beam axis) — needed by the new IDD Curve Distance metric, which has to measure depth along the exact same beam geometry the model was conditioned on.
4. Wrote `src/evaluation/metrics.py`:
   - `masked_mae()` — standalone (no-autograd) version of the official metric; matches `MaskedMAELoss` from Step 6 exactly.
   - `idd_curve_distance()` — bins dose by depth-along-beam-axis (using the new `beam_depth()`) via `scatter_add_`, then RMSD between predicted/GT depth-dose curves normalized by the GT curve's peak.
   - `stratified_plan_mae()` and `gamma_index_3d()` — implemented per the reference doc's definitions, but **both require a "prescription" dose that isn't in the local data** (see step 2). Both accept `prescription` as a parameter and documented the gap explicitly rather than hiding it; `evaluate.py` defaults to each patient's own full-plan max dose as an approximation.
   - `gamma_index_3d()`: local-normalization 3D gamma (window search sized to `distance_mm`/voxel spacing, minimum gamma over the window — the standard formulation), not a global-max-normalized approximation, since the doc explicitly says "local normalization."
   - `dvh_clinical_score()` — **deliberately returns `None` (not a fabricated number) when no `structures` dict is supplied**, since DoseRAD2026's local training data has no PTV/OAR contours at all. Implemented the actual D98%/V95%/D2%/Dmean computation so it's ready to use immediately if structures become available later (e.g. from the official evaluation container, or a future auto-segmentation step), but chose not to fake contours just to produce a number — a fabricated PTV would look like a real metric while being meaningless.
5. Wrote `src/evaluation/evaluate.py`: loads a checkpoint + config, rebuilds the same val-patient split used in training (via the new shared split helper — see below), runs the model over every held-out control point computing Level 1 metrics per-CP, accumulates each patient's full-plan dose (sum of that patient's control-point predictions across all 3 beams) for the Level 2 metrics, and prints a summary. Added `--cp_stride` and `--max_patients` flags for fast spot-checks during iteration, since a full validation pass over all held-out patients' 540 control points each would be slow to wait on repeatedly with ~16 days left.
6. **Caught and fixed a real bug while writing Step 7, in Step 6's code**: `scripts/train.py` built one `DoseRAD2026Dataset` instance and used `Subset` to carve out train vs. val indices — but both subsets shared the *same* underlying `transform`, which included `RandomIntensityShift` (an augmentation). That meant validation samples were also getting random noise added, contaminating val loss as a clean generalization signal. Fixed by building **two** separate `Dataset` instances (train transform includes the augmentation, val transform doesn't) and moved the train/val patient-ID split logic out of `scripts/train.py` into two new shared helpers in `src/data/hf_dataset.py` (`split_patient_ids()`, `indices_for_patients()`), so `evaluate.py` is guaranteed to reproduce the *exact* same held-out patients as training used, for the same config seed.
7. Smoke-tested every metric function individually on a real control point, then ran the full `evaluate()` pipeline against one real held-out patient (18 control points via `--cp_stride`-equivalent subsampling) on GPU. Caught and fixed one real bug during this: `dvh_clinical_score()`'s structure masks weren't squeezed to match the already-squeezed dose tensors, causing an `IndexError: too many indices for tensor of dimension 3` on the very first test with a (fake, sanity-check-only) structure mask — fixed by squeezing the mask in `_dvh_point_stats()` and in the PTV `V95` branch.

### Findings

- **Confirmed dataset-wide (not just spot-checked)**: no prescription dose or RTSTRUCT/segmentation data exists anywhere in the local `photon/training` download. `stratified_plan_mae`, `gamma_index_3d`, and `dvh_clinical_score` cannot be computed against true ground truth locally — only approximations (prescription≈plan max dose) or, for DVH, nothing at all. This should be treated as a standing limitation of local validation, not something to "fix" later without new data — worth checking the Grand Challenge platform/forum closer to the preliminary test-phase submission (Step 11) for whether structures are supplied by the official evaluation container.
- All metric values on the still-untrained model are large/nonsensical in absolute terms (e.g. masked_mae ~22,000-35,000, gamma pass rate 0.0%) — expected and consistent with the Step 6 finding that raw loss/metric magnitudes are large pre-training due to tiny absolute Gy values; these numbers are only meaningful as a trend once real training (Step 8) starts.
- Evaluating 18 real control points (one patient, GPU) took ~17s (~0.94s/CP) in this offline evaluation script — not a concern for the 1s/beam *inference* limit (Step 9 is a separate, more optimized code path), but confirms `--cp_stride` is worth using for quick iteration rather than always running a full validation pass.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/evaluation/metrics.py` | **New.** `masked_mae`, `idd_curve_distance`, `stratified_plan_mae`, `gamma_index_3d`, `dvh_clinical_score` (+ `_dvh_point_stats` helper). |
| `DoseRAD/src/evaluation/evaluate.py` | **New.** CLI: checkpoint + config → reproduces training's val split → runs Level 1 + Level 2 metrics, prints summary. |
| `DoseRAD/src/data/beam_encoder.py` | Refactored: extracted `_beam_geometry()`, added public `beam_depth()` method (reused by IDD metric). No behavior change to `encode()`. |
| `DoseRAD/src/data/hf_dataset.py` | Added `split_patient_ids()` and `indices_for_patients()` module-level helpers (moved out of `scripts/train.py`) so train/eval scripts share one split implementation. |
| `DoseRAD/scripts/train.py` | **Bug fix.** Now builds separate train/val `Dataset` instances so `RandomIntensityShift` augmentation no longer leaks into validation samples; uses the new shared split helpers. |
| `DoseRAD/CHECKLIST.md` | Marked Step 7 complete. |

### Checklist status after this session

- [x] **Step 7 — Evaluation metrics**: Level 1 metrics (Masked MAE, IDD Curve Distance) fully implemented and correct. Level 2 metrics (Stratified Plan MAE, Gamma Index) implemented but run on an approximated prescription value, documented as a known local-data limitation. DVH-Based Clinical Score implemented but returns `None` locally (no PTV/OAR contours available) rather than a fabricated number.

### Open questions / next steps

- Per `CHECKLIST.md`, next is **Step 8 — first real training run**: launch `scripts/train.py` against the full 77-patient dataset, watch training loss and `evaluate.py`'s Level 1/2 metrics trend on the held-out patients, and iterate on architecture/hyperparameters before shifting focus to speed optimization (Step 9). Given the ~16 days left, worth timing a single real epoch on the full data early to sanity-check the training-time budget before committing to the config's default 50 epochs.
- Revisit whether the official Grand Challenge evaluation supplies prescription/structure data that we don't currently have access to — would let `stratified_plan_mae`/`gamma_index_3d` use true values and `dvh_clinical_score` actually run, instead of the local approximations/gap documented above.

---

## Session: 2026-07-30 — Phase B, Step 6: Training loop & losses

### Steps taken, in order

1. Read the existing empty stubs (`src/training/losses.py`, `src/training/trainer.py`, `scripts/train.py`, `configs/task1_photon_ct.yaml`) and re-checked `DoseRAD2026_Complete_Reference.md`'s exact Masked MAE definition ("MAE in the high-dose region (>=10% of beam max), normalized by beam max") to make the training loss match what's actually scored, rather than picking an arbitrary generic loss.
2. Wrote `src/training/losses.py`:
   - `MaskedMAELoss`: differentiable version of the official Masked MAE metric itself (per-sample high-dose mask, normalized by that sample's beam max, averaged over the batch).
   - `BodyMaskedL1Loss`: auxiliary full-body-field L1 term (using the `body_mask` transform already built in Step 3), added because `MaskedMAELoss` alone gives zero gradient outside the high-dose core — without some full-field term the low/mid-dose penumbra would be free to be arbitrary, which wouldn't show up in Masked MAE but would hurt the other official metrics (Gamma Index, DVH-based Clinical Score) that do look at the whole distribution.
   - `DoseLoss`: weighted combination of the two (`w_masked=1.0`, `w_body=0.2` by default — metric-aligned term dominates).
3. Wrote `src/training/trainer.py` (`Trainer` class): train/validate loop, AMP (mixed precision) via `torch.autocast`/`torch.amp.GradScaler`, gradient clipping, optional gradient accumulation, checkpoint save (`last.pt` + `best.pt`) and resume.
   - **Deliberately fixed batch size at 1** (real DataLoader batch, not gradient accumulation): `PhotonBeamEncoder` isn't vectorized across samples with different CT origins, and the encoder has to be called once per control point regardless, so batching wouldn't save the expensive part. Documented this reasoning directly in the class docstring rather than leaving it implicit.
   - **Caches one `PhotonBeamEncoder` per patient** (`self._encoder_cache`, keyed by `patient_id`) inside the trainer, reused across that patient's control points *and* across epochs — this is what actually captures the ~78x GPU speedup measured in the Step 5 session; without the cache, every single training step would pay the coordinate-grid rebuild cost again.
4. Wrote `configs/task1_photon_ct.yaml`: data/model/loss/optimizer hyperparameters in one place (target shape/spacing, val_fraction, lr=2e-4, AdamW, cosine LR schedule, 50 epochs default, AMP on, grad_clip=1.0).
5. Wrote `scripts/train.py`: loads the config, builds the transform pipeline (same order as Step 3: Resample → CenterCropOrPad → BodyMask → NormalizeCT → RandomIntensityShift), builds the dataset, splits **by patient** (not by control point) into train/val, wires up model/loss/optimizer/scheduler, and calls `Trainer.fit()`. Supports `--resume <checkpoint>`.
   - **Patient-level split is deliberate**: a random per-control-point split would leak the same patient's anatomy into both train and val (one CT is shared by 540 control points), making val loss a misleadingly optimistic generalization signal. Documented in the function's docstring.
6. Smoke-tested the entire pipeline end-to-end on 3 real train / 2 real val control points (not the full 40,500 — that's Step 8's job) on GPU: forward pass, masked-MAE + body-L1 loss computation, backward pass with AMP + gradient clipping, optimizer step, LR scheduler step, checkpoint save, and checkpoint reload all completed without errors across 2 epochs.

### Findings

- Loss values at initialization are large (~26,000–29,000), **not a bug**: per-CP dose peaks are tiny in absolute terms (~1e-5 to 8.9e-5 Gy, per the Step 2 session's findings), so an untrained model's essentially-random output divided by that tiny `beam_max` denominator produces a large number. Gradient clipping (`grad_clip=1.0`) controls the actual optimization step size regardless, so this shouldn't destabilize training, but the raw loss number itself won't be very interpretable until the model starts learning to output near-zero baseline values — worth tracking whether it actually trends down once Step 8's real run starts, rather than assuming it will.
- Saw a standard PyTorch `UserWarning` ("Detected call of `lr_scheduler.step()` before `optimizer.step()`") on the very first smoke-test epoch — this is expected/benign AMP `GradScaler` behavior (it can skip an early optimizer step while calibrating the loss-scale factor), not a bug in the trainer.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/training/losses.py` | **New.** `MaskedMAELoss`, `BodyMaskedL1Loss`, `DoseLoss`. |
| `DoseRAD/src/training/trainer.py` | **New.** `Trainer` class — train/val loop, AMP, grad clipping/accumulation, per-patient beam-encoder cache, checkpointing. |
| `DoseRAD/configs/task1_photon_ct.yaml` | **New.** All data/model/loss/optimizer hyperparameters for Task 1. |
| `DoseRAD/scripts/train.py` | **New.** CLI entrypoint: config → dataset → patient-level train/val split → `Trainer.fit()`. |
| `DoseRAD/CHECKLIST.md` | Marked Steps 5 and 6 complete. |

### Checklist status after this session

- [x] **Step 6 — Training loop & losses**: implemented and smoke-tested end-to-end on real data (small subset) on GPU — loss computation, backward pass, optimizer/scheduler step, checkpoint save/reload all verified working.

### Open questions / next steps

- Per `CHECKLIST.md`, next is **Step 7 — evaluation metrics**: `src/evaluation/metrics.py` (Masked MAE, IDD Curve Distance, Stratified Plan MAE, 3D Local Gamma Index 1%/1mm, DVH-based Clinical Score) and `src/evaluation/evaluate.py`, so Step 8's real training run has proper validation metrics to watch beyond just the training loss.
- Not yet run on the full 40,500-sample dataset — the smoke test only proves the pipeline is *correct*, not how long a real epoch takes or whether the current batch-size-1/no-multi-GPU setup is fast enough to get through meaningful training within the ~16 days left before the deadline. Worth timing one real epoch (or a fixed number of steps) on the full data before committing to the 50-epoch default in the config.

---

## Session: 2026-07-29 — Full dataset download completed; Phase B, Step 5: UNet3D model + GPU beam-encoder fix

### Steps taken, in order

1. Kicked off the full `photon/training` download in the background (`nohup python scripts/download_data.py --task photon --split training --out data/photon_ct & disown`), decoupled from the terminal session so it survives independent of any single conversation.
2. Discovered an NVIDIA A100-PCIE-40GB GPU is available on this server (previously unconfirmed) — changes the speed-optimization strategy for Step 9, since GPU offload is on the table for the whole pipeline, not just the model.
3. Installed `monai` (3D UNet implementation) into `.venv`.
4. Wrote `src/models/unet3d.py`: `build_unet3d()`, a thin wrapper around MONAI's `UNet` (5-level encoder/decoder, channels `16→256`, instance norm, 2 residual units per level).
5. Wrote `src/models/beam_cond.py`: `BeamConditionedInput`, concatenates the normalized CT channel with the beam-encoder's aperture mask channel (`NUM_CHANNELS = 2`) along the channel dim. Documented why `body_mask` is deliberately *not* a model input (it's for loss masking later in `src/training/losses.py` instead — air/tissue is already visible in the CT channel itself).
6. Wrote `src/models/model_factory.py`: `DoseRADModel` (wires `BeamConditionedInput` → `UNet3D`) and a `build_model(config)` factory function.
7. Smoke-tested the full pipeline end-to-end on GPU (dataset → transforms → beam encoder → model): 4.8M params, output shape `(1,1,256,256,256)` correctly matches the dose target shape, no NaNs, warmed-up forward pass **0.074s**, peak GPU memory **1.09GB**. Well inside the 1s/beam budget on its own.
8. While smoke-testing, proactively timed the beam encoder itself (not just the model) since it's also on the per-beam critical path: **~0.73s/call on CPU**, close enough to the 1s/beam hard limit to be a real risk once other overhead (I/O, transforms, model) is added on top.
9. Tried moving the beam encoder's math to GPU as a cheap fix — hit `RuntimeError: Expected all tensors to be on the same device` because several tensors inside `encode()` (`iso`, `beam_axis`, `leaf_motion_axis`, `leaf_stack_axis`) were being created without an explicit `device=` argument, defaulting to CPU while the precomputed coordinate grid had been moved to `cuda`.
10. Fixed it: `encode()` now reads `device = self._coords_xyz.device` and passes `device=device` to every tensor it constructs, so it transparently follows whichever device the encoder instance's coordinate grid lives on.
11. Re-ran the timing test with the fix in place, comparing CPU vs GPU and checking the two versions still agree.

### Findings

- **Full dataset download is complete**: `data/photon_ct/photon/training/` contains all **77 patients**, **384GB** on disk. Background process is no longer running (finished cleanly, no crash).
- **GPU beam-encoder fix confirmed working**: CPU warm avg **0.630s/call** vs GPU warm avg **0.0081s/call** — a **~78x speedup**. CPU and GPU masks agree on **100%** of voxels (fix didn't change any math, just where it runs).
- **Net effect on the 1s/beam budget**: with both the model (0.074s) and the beam encoder (0.008s) on GPU, the two heaviest known per-beam computations together use under 0.1s, leaving substantial headroom for I/O, transforms, and pipeline overhead. The CPU-bound beam encoder is no longer a credible risk to the runtime limit, *provided* the inference pipeline (Step 9) actually keeps it on GPU end-to-end (encoder instantiated once per patient, coordinate grid moved to `cuda` immediately, all per-CP calls reusing it).

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/models/unet3d.py` | **New.** `build_unet3d()` — MONAI 3D UNet wrapper. |
| `DoseRAD/src/models/beam_cond.py` | **New.** `BeamConditionedInput` — concatenates CT + beam mask into model input channels. |
| `DoseRAD/src/models/model_factory.py` | **New.** `DoseRADModel` + `build_model()` — wires the above into the full model. |
| `DoseRAD/src/data/beam_encoder.py` | **Fixed.** `encode()` now creates all internal tensors on `self._coords_xyz.device` instead of implicitly on CPU, fixing a device-mismatch crash and enabling the ~78x GPU speedup. |
| `data/photon_ct/photon/training/` | Full download completed: 77 patients, 384GB. |

### Checklist status after this session

- [x] **Step 5 — Model architecture (UNet3D + beam conditioning)**: implemented and smoke-tested end-to-end on GPU with real data (CT + beam encoder → model → correctly-shaped dose prediction, no NaNs).
- Speed risk flagged in the Step 4 session (beam encoder CPU cost) is now resolved *for GPU inference*; still needs to be validated as part of Step 9's actual inference pipeline, not just this isolated test.

### Open questions / next steps

- Per `CHECKLIST.md`, next up is **Phase B, Step 6 — training loop & losses**: `src/training/losses.py` (masked/weighted loss matching the challenge's Masked MAE metric), `src/training/trainer.py` (loop, checkpointing, logging), `scripts/train.py`, `configs/task1_photon_ct.yaml` — all currently empty stubs.
- Then **Step 7 — evaluation metrics** (`src/evaluation/metrics.py`: Masked MAE, IDD Curve Distance, Stratified Plan MAE, 3D Local Gamma Index 1%/1mm, DVH-based Clinical Score) before a first real training run (Step 8).
- With the full 77-patient dataset now local, worth revisiting the beam-encoder `rotation_sign=-1` calibration (done on only 2 patients so far) against a larger sample once time allows — not urgent, but cheap insurance before final submission.

---

## Session: 2026-07-20 (continued) — Phase A, Step 4: Photon beam encoder

### Steps taken, in order

1. Read `src/data/beam_encoder.py`'s existing skeleton — confirmed `PhotonBeamEncoder.encode()` already expected roughly one-control-point-at-a-time input, consistent with the per-CP granularity confirmed earlier, but used stale key names (`isocenter`, `mlc_positions`) that don't match what `hf_dataset.py` actually produces (`iso_center`, `mlc_left_int_mm`/`mlc_right_int_mm`).
2. **Tried to empirically validate the gantry-rotation coordinate convention** before writing any geometry code, since the JSON gives no explicit IEC convention, couch angle, or collimator angle. First attempt: dose-weighted centroid of each control point's nonzero dose vs. isocenter, checked against candidate rotation directions. **This was inconclusive** — direction estimates were noisy and inconsistent across angles (e.g. z-component of the "direction" swung from ~0 to ~-0.96 between adjacent test angles), because a large fraction of the volume (~18%) has some nonzero scattered dose, swamping the geometric signal.
3. Implemented `PhotonBeamEncoder.encode()` for real: builds a divergent (conical) beam mask from a point source through an MLC aperture, using an IEC 61217-style model — gantry rotates about the patient's S-I axis (array z) at fixed z, sweeping the source through the axial (x,y) plane; MLC leaf-stacking axis stays fixed along z (collimator angle assumed 0, not present in the JSON); `mlc_left_int_mm`/`mlc_right_int_mm` define the aperture at the isocenter plane and scale linearly with depth (beam divergence). Precomputes the volume's physical coordinate grid once per encoder instance (reused across all of a patient's control points) for speed. Fixed the encoder's expected key names to match `hf_dataset.py`'s actual `beam_params` dict.
4. **Switched to a much more reliable validation method**: instead of a noisy centroid, directly measured what fraction of a control point's *real* dose falls inside vs. outside the predicted aperture mask (`dose[mask].sum() / dose.sum()`), for two candidate rotation-sign conventions (`rotation_sign = +1` vs `-1`).
5. Ran this on 8 spot-check angles of beam 0 first: found overlap was near-zero for both signs on control points with a genuinely narrow/near-closed MLC aperture (confirmed by directly inspecting `mlc_left/right` arrays: only 4-6 of 80 leaves open, ~2mm average width) — expected, not a bug, since body-wide scatter dominates the total-dose denominator for a sliver-thin beam. For wide-aperture control points, `rotation_sign=-1` consistently beat `+1` (e.g. 0.578 vs 0.415 overlap at gantry=134°).
6. Broadened the check to all 180 control points of beam 0 (filtering to the 106 with >15mm average aperture width to avoid the noisy narrow-field cases): **`rotation_sign=-1` won 100/106, `+1` won 4, 2 ties.**
7. Broadened further to all 6 beams across both downloaded patients (`1ABB006` beams 1-2, `1ABB011` beams 0-2; 600 wide-aperture control points total): **`rotation_sign=-1` won 553/600, `+1` won only 11, 36 ties.** For `1ABB011` specifically, `+1` won zero control points across all 3 beams. This is strong, reproducible evidence across patients and beams, not a fluke of one beam's geometry.
8. Locked in `rotation_sign=-1.0` as the default and documented the full calibration methodology and results directly in `beam_encoder.py`'s docstring, including the explicit caveat that this is a coarse geometric prior (no penumbra/scatter modeling) rather than dose-accurate ground truth.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/data/beam_encoder.py` | **Rewritten.** `PhotonBeamEncoder.__init__` now also takes `origin`; `encode()` implements the real divergent-beam/MLC-aperture geometry (was a zero-mask stub); key names aligned with `hf_dataset.py`'s actual `beam_params` dict. `ProtonBeamEncoder` untouched (out of scope — Task 1 is photon/CT). |

### Checklist status after this session

- [x] **Step 4 — Photon beam encoder**: implemented and empirically calibrated against real dose data (rotation-sign convention). Known limitation: coarse geometric mask only (no penumbra/scatter, no MLC transmission/leakage modeling) — documented in the code as a deliberate simplification, matching the original stub's "simplified cylindrical/conical mask" ambition rather than full dose-accurate ray-tracing.

### Open questions / next steps

- Validation so far only used the 2 downloaded patients. Worth a spot-check against a few more of the 77 once more patients are downloaded, to make sure `rotation_sign=-1` holds dataset-wide and isn't specific to these two.
- `beam_encoder.py` is not yet wired into `hf_dataset.py` or a collate function — that integration (plus per-patient caching of the encoder instance, since rebuilding its precomputed coordinate grid per-sample would be slow) is future work, likely alongside Phase B's model/training setup.
- Next per `CHECKLIST.md`: Phase B — Step 5 (UNet3D model + beam conditioning) and Step 6 (training loop & losses).

---

## Session: 2026-07-20

**Checklist item in progress:** Phase A, Step 2 — Explore the raw data

### Steps taken, in order

1. Installed `SimpleITK` and `numpy` into `.venv` (not previously installed — needed to read `.mha` volumes).
2. Wrote `scripts/explore_data.py`: loads a patient's CT, MR, a couple of dose `.mha` files, and the beam JSON, and prints shape/spacing/origin/value-range/structure for each.
3. Ran it against patient `1ABB006` and recorded findings (below).
4. Loaded all 180 control-point dose files for beam 0 and summed them to check whether individual CP doses are fragments of a larger per-beam dose (they are).
5. Cross-checked `DoseRAD2026_Complete_Reference.md` to confirm units/normalization and how beams are evaluated.

### Findings

- **CT / MR / Dose grids are all co-registered**: same size `(249, 246, 246)` (x,y,z), same spacing `2.0×2.0×2.0mm`, same origin — no resampling needed between them for this patient.
- **CT value range**: `[-1024, 2834]` (Hounsfield units, as expected). **MR value range**: `[0, 421]`.
- **Beam JSON structure** (`<PATIENT_ID>.json`): top-level `{"beams": [...]}`. Each beam has `beam_idx`, `SAD` (source-axis distance, 1000mm), `iso_center` (x,y,z in mm), `num_mlc_leaf_pairs` (80), and a `control_points` list. Each control point has `cp_idx`, `gantry_angle` (degrees), `mlc_left_int_mm` / `mlc_right_int_mm` (80-length arrays of MLC leaf positions in mm). **No explicit MU/weight field per control point.**
- Patient `1ABB006` has **3 beams, 180 control points each = 540 dose files**, matching the file count on disk. Gantry angle sweeps from -180° across control points — this is a **VMAT arc plan**, not static-gantry IMRT.
- **Individual per-CP dose files have tiny values** (max ~`8.9e-5` Gy) — summing all 180 CPs for beam 0 gives a per-beam max of only ~`0.0047` Gy, still small in absolute terms.
- Per `DoseRAD2026_Complete_Reference.md`: output is defined as **"beam-specific 3D dose distribution (Gy, dose-to-medium)"**, and **each beam is evaluated independently**. Combined with the summing test above, this strongly implies the real prediction target per beam is the **sum of that beam's per-CP dose files**, not the raw per-CP grids. Official metrics (Masked MAE, Gamma Index) are normalized by beam max / local dose, so the small absolute magnitude is expected and not a units bug.

> **⚠️ CORRECTION (later same session):** The bullet above is **wrong** — see below. Left in place rather than deleted so the log stays an honest record.

6. Re-read `DoseRAD2026_Complete_Reference.md` in full (had only skimmed it before) and found the "Data geometry notes" section: *"Each patient has multiple independent beams — each evaluated separately. Photon: per control point (VMAT arc sampling)."* It also states **40,500 photon beam segments** in the training set.
7. Checked the math: 75 training patients × 540 segments/patient (3 beams × 180 CPs, matching what we found on `1ABB006`) = **40,500 — exact match**. This confirms each individual control point is its own independent "beam segment" for training/evaluation purposes. **The per-CP dose files are NOT meant to be summed** — each one is a standalone target.
8. Checked `src/data/beam_encoder.py`'s existing skeleton (`PhotonBeamEncoder.encode(beam_json)`) — it already expects a single control point's params (`gantry_angle`, `isocenter`, MLC positions) and returns one spatial mask, consistent with the per-control-point granularity now confirmed. Noted a minor key-naming mismatch to fix later in Step 4: the stub reads `isocenter`/`mlc_positions`, but the real JSON uses `iso_center`/`mlc_left_int_mm`+`mlc_right_int_mm` — not fixed now, out of scope for the data loader.
9. Wrote `src/data/hf_dataset.py` as a local-disk PyTorch `Dataset`: one sample = one `(patient, beam, control_point)` triple → `{ct, dose, beam_params}`. Indexes all patients/beams/CPs under `data_root/task/split/` at init time, and caches up to 4 patients' CT volumes in memory (`functools.lru_cache`) to avoid re-reading the same ~57MB CT array from disk on every sample.
10. Installed `torch` into `.venv` (was in `requirements.txt` but not yet installed).
11. Smoke-tested `DoseRAD2026Dataset` against the 2 downloaded patients — see results below.

### Smoke test results

- `len(dataset)` = **1080** = 540 segments × 2 patients, as expected.
- `sample["ct"].shape` = `(1, 246, 246, 249)`, `sample["dose"].shape` = same — correct.
- `beam_params`: `gantry_angle` scalar, `iso_center` shape `(3,)`, `SAD` scalar, `mlc_left_int_mm`/`mlc_right_int_mm` shape `(80,)` each — matches JSON structure.
- Both `1ABB006` and `1ABB011` present in the index; last sample correctly resolves to `1ABB011`.

### Corrected findings

- **Prediction target = per-control-point dose, not summed per beam.** Confirmed via the 40,500 = 75 × 540 match plus the reference doc explicitly saying photon beams are evaluated "per control point." Session's earlier "sum over CPs" theory (bullet above) is superseded.

### Files created / changed

| Path | Change |
|---|---|
| `.venv` | Added `SimpleITK`, `numpy` packages |
| `DoseRAD/scripts/explore_data.py` | **New file.** One-off exploration script for CT/MR/dose/JSON of a given patient. |
| `DoseRAD/src/data/hf_dataset.py` | **Written** (was empty stub). Local-disk `DoseRAD2026Dataset(torch.utils.data.Dataset)`, one sample per control point. |

### Checklist status after this session

- [x] **Step 2 — Explore the raw data**: done. Shapes, spacing, value ranges, and JSON structure all understood; corrected per-control-point target granularity confirmed against the reference doc's segment count.
- [x] **Step 1 — Local data pipeline**: `src/data/hf_dataset.py` written and smoke-tested against the 2 sample patients.

### Open questions / next steps

- Step 4: implement `PhotonBeamEncoder.encode()` for real (currently a zero-mask stub), and align its expected key names (`isocenter`→`iso_center`, `mlc_positions`→`mlc_left_int_mm`/`mlc_right_int_mm`) with what `hf_dataset.py` actually returns.

---

## Session: 2026-07-20 (continued) — Phase A, Step 3: Preprocessing / transforms

### Steps taken, in order

1. Checked both downloaded patients' CT spacing directly — confirmed both already at `2×2×2mm` (matches the challenge's own documented pre-resampling), **but found their array sizes differ**: `1ABB006` is `(246,246,249)` (z,y,x), `1ABB011` is `(141,246,248)` — different anatomical extent (e.g. thoracic vs. abdominal coverage). This means PyTorch's default batching would fail (can't stack differently-shaped tensors) unless volumes are standardized to a common size — a real, concrete problem, not a hypothetical.
2. Extended `hf_dataset.py` to also return `ct_spacing` and `ct_origin` (physical-space metadata) per sample — needed so transforms can act on real geometry and so Step 4's beam ray-tracing can later map `iso_center` (physical mm) back into array indices correctly.
3. Wrote `src/data/transforms.py` with: `Compose`, `ResampleToSpacing` (no-op safety net — not expected to trigger given finding #1, but real trilinear resampling if it ever does), `CenterCropOrPad` (standardizes every sample to a fixed shape, e.g. `256×256×256`, for batching — updates `ct_origin` to keep physical alignment correct after the crop/pad), `NormalizeCT` (clip to `[-1000,3000]` HU, scale to `[0,1]`), `BodyMask` (simple HU-threshold body/air mask, not full segmentation), `RandomIntensityShift` (small additive Gaussian noise on normalized CT).
4. **Deliberately did not implement geometric augmentation** (flip/rotate) — flipping the CT/dose array without also transforming `beam_params` (gantry angle, iso_center, MLC positions — all real physical-space quantities) would silently decouple the input geometry from the target dose. Documented this as the reason only intensity-only augmentation is provided.
5. Smoke-tested the full `Compose` pipeline against both patients, **using the same transform order suggested in the module docstring** (`ResampleToSpacing → CenterCropOrPad → NormalizeCT → BodyMask → RandomIntensityShift`) — and caught a real bug: `body_mask` came out as 100% (`256³` — the entire padded volume, including background air, marked "body"). Root cause: `BodyMask` thresholds raw HU values (`> -500`), but running it *after* `NormalizeCT` meant it was thresholding already-rescaled `[0,1]` values instead, where every voxel (including padding) is `≥0 > -500`.
6. Fixed by reordering to `BodyMask` **before** `NormalizeCT` (also fixed the incorrect order shown in the module docstring, which had suggested the buggy order). Re-ran the test: `body_mask` now covers ~12-16% of the padded volume per patient — a plausible fraction for a real body inside a mostly-air-padded cube.
7. Re-ran a full `DataLoader(batch_size=4, shuffle=True)` pull mixing both patients — batched cleanly to `torch.Size([4, 1, 256, 256, 256])`.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/src/data/hf_dataset.py` | **Modified.** `_load_ct_uncached` now also returns spacing + origin; sample dict gains `ct_spacing`, `ct_origin`. |
| `DoseRAD/src/data/transforms.py` | **Written** (was empty stub). `Compose`, `ResampleToSpacing`, `CenterCropOrPad`, `NormalizeCT`, `BodyMask`, `RandomIntensityShift`. |

### Checklist status after this session

- [x] **Step 3 — Preprocessing/transforms**: done and smoke-tested, including a real batching test across two differently-shaped patients.

### Open questions / next steps

- `ResampleToSpacing` is untested against a real mismatched-spacing case — all patients seen so far are already `2×2×2mm`. Worth spot-checking a handful more of the 77 training patients before assuming this holds dataset-wide.
- `target_shape=(256,256,256)` for `CenterCropOrPad` was chosen generously to fit the two known patients; not verified against the full 77-patient size distribution — could crop a larger patient if one exists. Worth checking max dimensions across more patients before committing to this as final.
- Next: Step 4 — implement `PhotonBeamEncoder.encode()` for real, using `ct_spacing`/`ct_origin` now available from the dataset to correctly place `iso_center` and MLC aperture geometry in array space.

---

## Session: 2026-07-13

**Checklist item in progress:** Phase A, Step 1 — Local data pipeline (partially done, see status below)

### Steps taken, in order

1. Reviewed current repo state against prior session notes — confirmed all `src/` files still empty stubs except `src/data/beam_encoder.py` (skeleton with TODOs), `data/` folder empty.
2. Read `CHECKLIST.md` to confirm the 12-step build plan (Phase A: Data, Phase B: Model & Training, Phase C: Speed & Packaging, Phase D: Submission).
3. Discussed where the dataset should live: decided to download it fully to **this server** (has ~2TB free) rather than stream it, since streaming inside Colab was already ruled out in a prior session.
4. Decided to download **1-2 sample patients first** (not the full dataset) to validate the pipeline before committing to a long full download — full dataset is 864GB per the old (stale) Colab notes.
5. Logged into Hugging Face CLI using the token in `Hugging Face Token.txt` (token itself never printed/read into chat).
6. The dataset repo ID referenced in old notes (`doserad2026/dataset`) turned out to be a placeholder — it doesn't exist. Searched the HF Hub and found the real dataset: **`LMUK-RADONC-PHYS-RES/DoseRAD2026`**.
7. Inspected the real dataset's file structure via the HF API:
   - Top-level dirs: `photon/`, `proton/`
   - `photon/training/` contains **77 patients**
   - Each patient folder contains:
     - `<PATIENT_ID>.json` — beam configuration (~1.4MB)
     - `image/ct.mha` (~10MB), `image/mr.mha` (~5MB)
     - `dose/Dose_B{beam}_CP{control_point}.mha` — one file per beam/control-point combo (540 files for patient `1ABB006`, ~10MB each)
   - Estimated ~5.4GB per patient (dominated by the dose files).
8. Wrote a new script, `scripts/download_data.py`, using `huggingface_hub.snapshot_download` with `allow_patterns` so it pulls only the requested patients' folders instead of the whole 864GB repo. Takes `--patients`, `--task` (photon/proton), `--split` (training/testing), `--out` args.
9. Ran the script for patients `1ABB006` and `1ABB011` (same two used in an earlier session) → **9.0GB downloaded** in ~12 minutes.

### Files created / changed

| Path | Change |
|---|---|
| `DoseRAD/scripts/download_data.py` | **New file.** Downloads specific patients from the HF dataset repo to local disk. |
| `data/photon_ct/photon/training/1ABB006/` | **New.** `1ABB006.json`, `image/ct.mha`, `image/mr.mha`, `dose/*.mha` (540 files) |
| `data/photon_ct/photon/training/1ABB011/` | **New.** Same structure as above, for patient `1ABB011`. |
| `data/photon_ct/.cache/huggingface/` | **New.** HF's own download bookkeeping (safe to gitignore). |
| `~/.cache/huggingface/token` | HF auth token stored locally by `hf auth login` (outside the repo). |

### Decisions made

- **Confirmed dataset source:** `LMUK-RADONC-PHYS-RES/DoseRAD2026` on Hugging Face (supersedes the placeholder repo ID in the archived Colab notes — that doc is stale on this point too).
- Data will live at `data/photon_ct/` on this machine going forward.

### Checklist status after this session

- [~] **Step 1 — Local data pipeline**: download script done and sample data on disk; `src/data/hf_dataset.py` (the local-disk PyTorch `Dataset` loader) **still empty, not yet written**.
- [ ] **Step 2 — Explore the raw data**: not started. Natural next step — load the CT/dose/JSON for one patient with SimpleITK/numpy to check shapes, spacing, and value ranges before writing preprocessing or the loader.

### Open questions / next steps

- Explore raw data (Checklist Step 2) before writing `hf_dataset.py`, so the loader is built against real shapes rather than assumptions.
- Once validated on the sample, queue the full dataset download in the background (864GB, will take a while).
