# DoseRAD2026 — Complete Challenge Reference

> **Real-time Photon and Proton Dose Calculation on CT and MRI**
> Platform: [doserad2026.grand-challenge.org](https://doserad2026.grand-challenge.org)

---

## 🎯 Challenge Overview

DoseRAD2026 benchmarks **fast and accurate 3D radiation dose calculation** algorithms for both photon and proton radiotherapy, using CT or MRI as input. The ground truth is high-fidelity **Monte Carlo (Geant4) simulations**.

**Core I/O:**
- **Input**: 3D patient CT or MRI volume + beam delivery parameters
- **Output**: Beam-specific 3D dose distribution (Gy, dose-to-medium)

---

## 🧪 Four Tasks

| Task | Modality | Imaging | Clinical Use |
|------|----------|---------|--------------|
| 1 | Photon | CT | VMAT/IMRT — standard for most cancer patients |
| 2 | Photon | MRI | MRI-linac systems for online adaptive therapy |
| 3 | Proton | CT | High-precision proton therapy with Bragg peak |
| 4 | Proton | MRI | MRI-only proton workflows & future MRI-guided proton therapy |

All four tasks share the **same unified, spatially aligned MRI–CT dataset**.

---

## 📅 Timeline

| Milestone | Date |
|-----------|------|
| Website online / Training data release | 10/04/2026 |
| Training & Validation phase (18 weeks) | 10/04/2026 – 15/08/2026 |
| Introduced at ESTRO 2026 | 15/05/2026 – 19/05/2026 |
| **Preliminary test phase** (max 10 submissions) | 01/06/2026 – 15/08/2026 |
| **Final test phase** (max 2 submissions) | 16/07/2026 – 15/08/2026 |
| Winner announcements / invitations | 15/09/2026 |
| **Results presented at MICCAI 2026**, Strasbourg | 01/10/2026 |
| Post-challenge phase | 15/08/2026 – 31/12/2026 |

> ⚠️ **We are currently in the Preliminary Test Phase** (01/06 – 15/08/2026). Max 10 submissions allowed.

---

## 📦 Dataset

### Overview
- **First publicly available dataset** with paired CT + MRI + Monte Carlo beam-level doses
- **864 GB** total — available via Zenodo: [doi.org/10.5281/zenodo.19347848](https://doi.org/10.5281/zenodo.19347848)
- Dataset paper: [arXiv:2604.12778](https://doi.org/10.48550/arXiv.2604.12778)

### Cohort
| Split | Patients | Notes |
|-------|----------|-------|
| **Training** | 75 (36 abdominal + 39 thoracic) | Released April 2026 — full access |
| **Test** | 40 | Private until March 2030 |
| **External test** | 7 | Private, no public release ever |

Each case includes:
- Planning MRI (MR-Linac, **0.35T bSSFP**)
- Deformably registered CT
- Beam configuration data (JSON)
- Beam-level Monte Carlo dose distributions (.mha)

### Image Pre-processing
- Deformable CT→MRI registration
- Air cavity correction via MR-based segmentation
- Body masking and foreground standardization
- **Resampling**: Photon tasks → 2×2×2 mm³ | Proton tasks → 1×1×3 mm³

### Beam Configurations

**Photon Beams (VMAT-style)**
- Linac model: Elekta Versa HD with Agility 160-leaf MLC
- Defined by: MLC apertures, gantry angles, isocenter positions
- **40,500 photon beam segments** in training set

**Proton Beams (Pencil Beam Scanning)**
- Energy-dependent single-Gaussian approximation
- Energy range: **31.7 – 200.8 MeV**
- Defined by: source position, gantry angle, pencil beam spot grids
- **81,000 proton beamlets** in training set

### Ground Truth
- Full **Geant4 Monte Carlo** particle transport
- Tissue-dependent material modeling (CT-derived density maps)
- Dose-to-medium (Gy), beam-specific grids, masked to patient body

### License
**CC BY-NC 4.0** — Research/education use only, non-commercial, attribution required.

---

## 📊 Evaluation Metrics & Ranking

Algorithms are evaluated on **dose accuracy** and **computational efficiency**.

### Level 1 — Single-Beam Evaluation (per beam/segment)

| Metric | Description |
|--------|-------------|
| **Masked MAE** | MAE in the high-dose region (≥10% of beam max), normalized by beam max |
| **IDD Curve Distance** | RMSD of Integrated Depth-Dose curves, normalized by peak GT IDD |

### Level 2 — Full-Plan Evaluation (aggregated across beams with clinical weights)

| Metric | Description |
|--------|-------------|
| **Stratified Plan MAE** | MAE in 3 dose strata: High (≥80%), Mid (30–80%), Low (10–30%) of prescription |
| **3D Local Gamma Index (1%/1mm)** | % of voxels passing γ ≤ 1 with local normalization, in region ≥10% prescription |
| **DVH-Based Clinical Score** | For PTV: D98%, V95%; For 3 closest OARs: D2%, Dmean. Avg relative difference |

### Efficiency Metric
- **Average runtime per beam** = total time ÷ total beams
- Hardware: **AWS g5 GPU instance** (mandatory)
- ⚠️ **Hard limit: >1 second/beam = disqualified from official ranking**

### Ranking Method (RankThenMean)

1. Each submission ranked per metric (rank 1 = best)
2. Final score = **weighted average of per-metric ranks**:
   - Accuracy metrics: weight **1×** each (5 metrics)
   - Runtime: weight **2×** (double weight — real-time is critical)
3. Lowest final score wins (lower = better)

**Tie-breaking order:**
1. Lower runtime per beam
2. Lower combined plan-level MAE
3. Lower DVH score

---

## 📋 Rules Summary

| Rule | Detail |
|------|--------|
| **Account** | Must be verified Grand Challenge account |
| **Methods** | Fully automatic only |
| **Team size** | Max 5 participants |
| **Hardware** | AWS g5 GPU instance only |
| **Time limit** | ≤1 second per beam (hard limit) |
| **External data** | Allowed if publicly available before 10/04/2026 |
| **Pre-trained models** | Allowed if public before 10/04/2026 |
| **Private models/data** | NOT allowed |
| **Withdrawal** | Once submitted, cannot withdraw |

### Award Eligibility Requirements
- Present method in-person at **MICCAI 2026** (Strasbourg, 1 Oct 2026)
- Submit a paper in LNCS format describing the method
- Submit algorithm details form post-test
- **Top 4 teams must open-source code + weights within 14 days** of winner announcement
- Cite challenge report in future publications

---

## 🏆 Awards

**$1,000 USD per task** (4 tasks = $4,000 total prize pool)

Sponsored by: **Radformation** (New York, USA) and **RaySearch** (Stockholm, Sweden)

Results published in a **Medical Image Analysis** challenge paper (top teams invited as co-authors).

---

## 👥 Organizers

| Name | Institution |
|------|-------------|
| Adrian Thummerer | University of Bern, Switzerland |
| Christopher Kurz | LMU Klinikum, Munich, Germany |
| Fan Xiao | LMU Klinikum, Munich, Germany |
| George Dedes | Ludwig Maximilians University, Munich, Germany |
| Guillaume Landry | LMU Klinikum, Munich, Germany |
| Lennart Volz | GSI, Darmstadt, Germany |
| Matteo Maspero | UMC Utrecht, Netherlands |
| Miguel Palacios | Amsterdam UMC, Netherlands |
| Muheng Li | Paul Scherrer Institut PSI, Switzerland |
| Niklas Wahl | DKFZ, Heidelberg, Germany |
| Nikolaos Delopoulos | LMU Klinikum, Munich, Germany |
| Viktor Rogowski | Skane University Hospital, Lund, Sweden |
| Ye Zhang | Paul Scherrer Institut PSI, Switzerland |
| Zoltan Perko | TU Delft, Netherlands |

**Contact:** guillame.landry@med.uni-muenchen.de

**Endorsed by:** ESTRO, PTCOG, DGMP, EFOMP, EFIE, NVKF, SSRMP, SWEBMP

---

## 🔑 Key Technical Insights for Participation

### What to optimize for:
1. **Speed is heavily weighted** (2× in final rank) — must be ≤1 sec/beam on AWS g5
2. **Beam-level accuracy first** (MAE, IDD) — these directly reflect your core model quality
3. **Plan-level metrics** (Gamma, DVH) emerge from summing beams with clinical weights — no extra steps needed from participants

### Data geometry notes:
- Each patient has **multiple independent beams** — each evaluated separately
- Photon: per control point (VMAT arc sampling)
- Proton: per pencil beamlet
- Output must be resampled to the **reference dose grid** if your model uses a different internal resolution

### Submission format:
- Algorithm submitted as a **Docker container** on Grand Challenge
- Must run on **AWS g5 instance** (select in submission settings)
- Submission instructions are locked — must join the challenge first

---

## 🔗 Key Links

| Resource | URL |
|----------|-----|
| Challenge homepage | https://doserad2026.grand-challenge.org/ |
| Dataset (Zenodo, 864 GB) | https://doi.org/10.5281/zenodo.19347848 |
| Dataset paper (arXiv) | https://doi.org/10.48550/arXiv.2604.12778 |
| Challenge design doc | https://zenodo.org/records/19714006 |
| Forum | https://doserad2026.grand-challenge.org/forum/topics/ |
| Leaderboard — Photon/CT | https://doserad2026.grand-challenge.org/evaluation/photon-dose-preliminary-testing/leaderboard/ |
| Leaderboard — Photon/MR | https://doserad2026.grand-challenge.org/evaluation/proton-dose-preliminary-testing/leaderboard/ |
| Leaderboard — Proton/CT | https://doserad2026.grand-challenge.org/evaluation/proton-dose-on-ct-preliminary-testing/leaderboard/ |
| Leaderboard — Proton/MR | https://doserad2026.grand-challenge.org/evaluation/proton-dose-on-mr-preliminary-testing/leaderboard/ |
