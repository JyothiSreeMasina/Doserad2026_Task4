# DoseRAD2026 Challenge — Task Checklist

> Task 1 (Photon / CT) build plan. **Corrected 2026-08-14, verified directly on the live grand-challenge.org submission pages** (superseding earlier notes in this file, which wrongly said 2026-08-15): both the Preliminary and Final Testing Phase for Photon/CT **close 2026-08-31, 5:59pm America/New_York**. Preliminary allows 10 submissions, Final allows 2. Job timeout on the platform is 10 minutes/case (separate from the ≤1s/beam *scoring* criterion, which is a ranking factor not a hard kill-timer).
> See `DoseRAD2026_Complete_Reference.md` for full rules/metrics; see `archive/Colab_Workflow.md` for the old (now abandoned) streaming approach.

---

## Phase A — Data

- [x] **1. Local data pipeline** — Write a script to download the DoseRAD2026 Task 1 training data directly to local disk (no more Colab/HF streaming), and rewrite `src/data/hf_dataset.py` (currently emptied) as a local-disk PyTorch `Dataset` that reads CT images, dose `.mha` volumes, and beam JSON configs from disk.
- [x] **2. Explore the raw data** — Load one patient's CT image, dose volumes, and beam JSON with SimpleITK/numpy to understand shapes, spacing, value ranges, and how control points (`CP`) and beams (`B`) in filenames map to the JSON beam config, before writing preprocessing code.
- [x] **3. Preprocessing / transforms** — Fill in `src/data/transforms.py`: resample to the 2×2×2mm reference grid (photon tasks), normalize CT intensities, apply body masking, add any training augmentation.
- [x] **4. Photon beam encoder** — Finish `src/data/beam_encoder.py`'s `PhotonBeamEncoder.encode()` (currently a TODO stub returning a zero mask): implement ray-tracing/geometry from gantry angle, isocenter, and MLC aperture positions (Elekta Versa HD, Agility 160-leaf MLC) into a 3D beam path mask/feature map.

## Phase B — Model & Training

- [x] **5. UNet3D model** — Fill in `src/models/unet3d.py` (3D U-Net, likely via MONAI) and `src/models/beam_cond.py` (how beam features condition the network), wired together via `src/models/model_factory.py`.
- [x] **6. Training loop & losses** — Fill in `src/training/losses.py` (masked/weighted MAE matching the challenge metric) and `src/training/trainer.py` (loop, checkpointing, logging); write `scripts/train.py` and fill in `configs/task1_photon_ct.yaml`.
- [x] **7. Evaluation metrics** — Fill in `src/evaluation/metrics.py` implementing the official metrics (Masked MAE, IDD curve distance, Stratified Plan MAE, 3D Local Gamma Index 1%/1mm, DVH-based clinical score) and `src/evaluation/evaluate.py` to run them on validation data.
- [x] **8. First training run** — Training completed (10/10 epochs, 2026-08-07). Root-caused the outside-body dose leakage (see PROGRESS_LOG.md 2026-08-10), added `OutsideBodyL1Loss`, fine-tuned from epoch 4 for 5 more epochs (2026-08-12/14). Best checkpoint now epoch 5: masked_mae 9.5%, idd_curve_distance **0.56** (was 126.5), gamma_index_pass_rate 3.3% (was 1.3%). See PROGRESS_LOG.md 2026-08-14 entry.

## Phase C — Speed & Packaging

- [x] **9. Inference & speed optimization** — Fill in `scripts/inference.py`; profile and optimize (mixed precision, batching, avoiding redundant recompute) to hit the challenge's hard limit of **≤1 second per beam** on an AWS g5 GPU instance. Missing this disqualifies the submission from ranking regardless of accuracy.
- [x] **10. Docker submission container** — Built with the fixed epoch-5 checkpoint baked in, end-to-end tested (real patient in, correct output out, well under the 1s/beam limit), packaged via `do_save.sh` into `docker/doserad2026_task1_photon_ct_2026-08-14T10-18-36.670007874-04-00.tar.gz` (3.7G) + `docker/model.tar.gz` (51M). Ready to upload.

## Phase D — Submission

- [ ] **11. Preliminary test phase submission** — Verify Grand Challenge account, join the DoseRAD2026 challenge, follow submission instructions, submit the Docker container on an AWS g5 instance (up to 10 submissions allowed, through 2026-08-15). Review the leaderboard and iterate.
- [ ] **12. Final submission** — Submit the final entry (max 2 submissions, window 2026-07-16 to 2026-08-15). Hard deadline, cannot be withdrawn once sent.

---

## Key constraints to keep in mind throughout

- **Runtime hard limit:** ≤1 second/beam on AWS g5 GPU — disqualifies the submission from ranking if exceeded. Weighted **2×** in final score (heaviest single factor).
- **Fully automatic methods only** — no manual intervention allowed at inference time.
- **Private data/models not allowed** — only data/pretrained models public before 2026-04-10.
- Top 4 teams must open-source code + weights within 14 days of winner announcement, and present in person at MICCAI 2026 (Strasbourg, 1 Oct 2026) to be award-eligible.
