# DoseRAD2026 — Proton Dose Prediction on MRI

Submission code for the proton/MRI task of the
[DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/):
predicting a 3D radiation dose distribution (Geant4 Monte Carlo ground
truth) for a single pencil-beam-scanning (PBS) beamlet, directly from a
patient MRI volume and that beamlet's source, target, and energy.

This is the compositional task of the team's four submissions: the proton
beamlet encoder built for the [proton/CT
task](https://github.com/JyothiSreeMasina/Doserad2026-Task3), combined with
the MRI preprocessing built for the [photon/MR
task](https://github.com/JyothiSreeMasina/Doserad2026-Task2), reusing the
shared architecture with zero new source files. The full write-up —
training, a controlled with/without-correction comparison run before
submitting (rather than discovered after, as happened on proton/CT), and
packaging — is the companion LNCS report, [*A Physics-Conditioned 3D U-Net
for Proton Dose Prediction on MRI*](paper/proton_dose_mr_lncs.pdf), included
in this repository.

## Approach

The same 3D U-Net used across this team's four DoseRAD2026 submissions
(5-level encoder/decoder, 16–32–64–128–256 channels, two residual units per
block, ~4.8M parameters, built on [MONAI](https://monai.io/)'s `UNet`) takes
a two-channel input — a normalized MRI volume and a beamlet-conditioning
mask — and predicts a single-channel dose volume. The beamlet mask comes
from the analytic, first-principles `ProtonBeamEncoder`
(`src/data/beam_encoder.py`): range from the Bragg-Kleeman relation, range
straggling from Bortfeld's mono-energetic approximation, lateral spread from
the Highland multiple-Coulomb-scattering formula, and a
water-equivalent-path-length correction that samples MR-derived stopping
power along the beamlet axis. MRI preprocessing (per-volume 99th-percentile
normalization, intensity-thresholded body mask) is identical to the
photon/MR submission's.

Unlike the proton/CT submission, the same train/inference consideration
here — whether the deployed checkpoint should receive the physically
corrected WEPL input it never trained on — was caught and tested with a
controlled comparison *before* submitting, not discovered afterward; see
Section 3.2 of the paper for the result and the reasoning behind which
configuration was deployed.

## Layout

```
src/                              Data pipeline, beam encoder, model, losses, training loop, evaluation metrics
scripts/                          train.py, evaluate_cloud.py — training and evaluation entry points
configs/task4_proton_mr.yaml      Training config (beam type, modality, hyperparameters)
docker_task4_proton_mr/           app.py, process.py, Dockerfile, build/test scripts for the submission container
Dockerfile                        Root-level copy of the Dockerfile above (Grand Challenge's repo-linked build looks for ./Dockerfile with no configurable path)
paper/                            LNCS algorithm-description report for this task
```

## Reproducing

```bash
pip install -r docker_task4_proton_mr/requirements.txt
```

**Train:**
```bash
python scripts/train.py --config configs/task4_proton_mr.yaml
```

**Evaluate against local held-out patients:**
```bash
python scripts/evaluate_cloud.py --config configs/task4_proton_mr.yaml --checkpoint checkpoints/task4_proton_mr/best.pt
```

Training data (CT + MR + proton beam JSON + Geant4 beamlet-level dose,
shared with the proton/CT release) is released by the challenge organizers
on [Zenodo](https://doi.org/10.5281/zenodo.19347848) and is not included
here.

**Build and run the submission container:**
```bash
docker build --platform=linux/amd64 -f Dockerfile -t doserad2026_task4_proton_mr .
docker_task4_proton_mr/do_test_run.sh   # health check + one /invoke call against test fixtures
```
The container implements the platform's required `/health` + `/invoke` HTTP
API and the documented 10-slot batched image/metadata I/O contract.

## Weights

Trained checkpoints are not tracked in this repository. The one actually
submitted and scored is uploaded to the Grand Challenge platform separately
from the container image, per the platform's own `model.tar.gz` mechanism.

## License

CC BY-NC 4.0, matching the DoseRAD2026 dataset license.
