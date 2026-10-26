# Clinical pipeline demo

Simple website for live inference on the saved Engineering-thesis models.

Open http://127.0.0.1:8000 after:

```powershell
cd eeg_qgan
python demo/server.py
```

## What supervisors see

One clinical page:

1. Holdout ictal windows from TUH-SZ
2. EEG review canvas
3. **Run live comparison** across hybrid Style-qGAN, capacity-matched twin, and Gaussian mixture
4. Generation metrics computed now (latent MMD², sliced W₁, autocorrelation, spectral)
5. Detection metrics computed now (detector probabilities on real ictal clips and on a synthetic clip from each generator)

No recorded AUROC is shown as a live result. Full-holdout AUROC needs the TUH corpus on Narval.

## Models for this inference demo

From `eeg_qgan_results_thesis_figures/thesis_figure_capture/artifacts/seed0_fold0`:

- frozen autoencoder
- hybrid Style-qGAN (704 params)
- capacity-matched twin (709 params)
- Gaussian mixture
- four STFT–BiLSTM detectors (baseline + each generator)

Those are the only models this clinical demo needs. Mechanism studies
(entanglement topologies, Fourier, rematch extras) are generation-only on Narval
and are **not** part of this page.

If the clinical fold is missing:

```powershell
cd c:\Users\meriemterki\Documents\Thesis\eeg_qgan
.\demo\sync_from_narval.ps1
python demo/verify_studies.py
```

Do not mix older `paper_phase1` checkpoints into this demo: they sit on a different autoencoder.

## Main API

| Method | Path | Role |
|--------|------|------|
| POST | `/api/pipeline/live-compare` | Regenerate every arm and score generation + detection live |
| POST | `/api/pipeline/run` | One arm end to end |
| GET | `/api/pipeline` | Available models and stages |
| GET | `/docs` | OpenAPI |

## Verify

```powershell
python demo/verify.py
```
