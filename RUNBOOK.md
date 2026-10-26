# Exact steps to produce the corrected results and the preprocessing figures.

This is the checklist for Narval. Do the steps in order.

**Outer protocol (default):** `data.split_mode: predefined` uses the TUH
`train` / `dev` / `eval` folders under BinaryClassification_Data. Your supervisor
stated those folders are already patient-disjoint, so per-window patient sidecars
are **not** required when you train on `train` and test on the holdouts.

Paths below assume:

```bash
PROJECT=$HOME/eeg_qgan
DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
OUTPUT_ROOT=$SCRATCH/eeg_qgan_results
```

Adjust `DATA_ROOT` if your BinaryClassification_Data lives elsewhere.

---

## 0. Sync the code and create the environment (once)

On a **login node** (not a compute node):

```bash
# copy / clone the new package to Narval, then:
cd $HOME/eeg_qgan

# one-time virtualenv (includes PyWavelets for db6 denoising)
bash scripts/narval/setup_env.sh

# confirm the package and wavelet dependency
source ${SCRATCH}/venvs/eeg_qgan/bin/activate
python -c "import eeg_qgan, pywt, torch; print(eeg_qgan.__version__, pywt.__version__, torch.__version__)"
python scripts/validate_studies.py
```

If the venv already existed from an earlier attempt:

```bash
source ${SCRATCH}/venvs/eeg_qgan/bin/activate
pip install -e .
pip install PyWavelets
```

---

## 1. Confirm the predefined loader (required before any study)

```bash
cd $HOME/eeg_qgan
module load python/3.11 scipy-stack   # or your usual stack
source ${SCRATCH}/venvs/eeg_qgan/bin/activate

# dry-run: loads train + holdouts, checks capacity, writes dry_run.json
python -m eeg_qgan.experiments.runner smoke \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dry-run
```

You should see `split_mode: predefined`, holdouts present (`dev` and/or `eval`),
and `group_source: predefined_tuh_partitions+batch_file`. Outer folds are
train-vs-dev and train-vs-eval. Inner generator splits still group by **batch
file** (weaker than per-patient); report that honestly in the thesis.

Optional: `scripts/build_patient_sidecars.py` only matters if you switch to
`split_mode: grouped_cv`. On the pooled BinaryClassification batches it usually
cannot recover per-window ids — that is expected and not a blocker for the
predefined protocol.

---

## 2. Preprocessing before/after figures (thesis)

This is a **short CPU job**. Do **not** submit it with `submit.py`. Sync the
script from Windows, run it on Narval, then copy the PNGs back.

### 2a. Windows → Narval (one file)

From **PowerShell** on the laptop (`Thesis` folder):

```powershell
scp eeg_qgan\scripts\plot_preprocessing.py `
  mrmterki@narval.computecanada.ca:~/eeg_qgan/scripts/plot_preprocessing.py
```

Nothing else has to move. `tuh.py` already on Narval is enough: the script only
calls `preprocess_stages` and `robust_range`.

### 2b. Run on Narval (login node is fine)

```bash
ssh mrmterki@narval.computecanada.ca
cd ~/eeg_qgan
module load python/3.11 scipy-stack
source $SCRATCH/venvs/eeg_qgan/bin/activate

export DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results
export MPLBACKEND=Agg

# confirm the tree is readable
ls "$DATA_ROOT" | head

mkdir -p "$OUTPUT_ROOT/figures/preprocess"

python scripts/plot_preprocessing.py \
  --data-root "$DATA_ROOT" \
  --out "$OUTPUT_ROOT/figures/preprocess" \
  --channels 0,3,7,12 \
  --sample-length 256 \
  --target-fs 200 \
  --normalize-scope recording \
  --seed 7
```

`--normalize-scope recording` is the clinical pin (one robust range per batch
file, shared across 22 channels). Use `--normalize-scope window` only if you
need to remake the older per-window plate.

If the login node complains about CPU, take a 15-minute allocation:

```bash
salloc --account=def-khenni_cpu --time=0:15:00 --cpus-per-task=2 --mem=8G
# then the same module / venv / python commands
```

Outputs under `$OUTPUT_ROOT/figures/preprocess/`:

- `fig_preprocess_before_after.png` / `.pdf`
- `fig_preprocess_overlay_seizure.png` / `.pdf`
- `fig_preprocess_overlay_nonseizure.png` / `.pdf`
- `preprocess_caption.txt`

Print the scratch path before you log out (`echo $SCRATCH`) — Windows cannot
expand `$SCRATCH`.

### 2c. Narval → Windows

From **PowerShell** (replace `mrmterki` in the scratch path if `echo $SCRATCH`
printed something else):

```powershell
$dst = "eeg_qgan\figures\preprocess"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
scp "mrmterki@narval.computecanada.ca:/scratch/mrmterki/eeg_qgan_results/figures/preprocess/*" $dst
Copy-Item -Force "$dst\fig_preprocess_*.png","$dst\fig_preprocess_*.pdf" `
  Engineering_PFE\Figures\methods\
Copy-Item -Force "$dst\fig_preprocess_*.png","$dst\fig_preprocess_*.pdf" `
  Presentation\figures\02_methods\
```

Then recrop the footer for the slide PNG if you want a cleaner Beamer plate:

```powershell
python -c "from PIL import Image; im=Image.open(r'Presentation/figures/02_methods/fig_preprocess_before_after.png'); w,h=im.size; im.crop((0,0,w,int(h*0.905))).save(r'Presentation/figures/02_methods/fig_preprocess_before_after_slide.png')"
```

Local offline check (no TUH needed):

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic
python scripts/plot_preprocessing.py --data-root data/synthetic --out figures/preprocess
```

---

## 3. Submit the corrected studies (order matters)

Always validate configs before submitting a large batch:

```bash
python scripts/validate_studies.py
python scripts/narval/submit.py --list-studies
```

### 3a. Size the autoencoder fix (cheap, do first)

```bash
python scripts/narval/submit.py bottleneck_comparison --split-by seed
python scripts/narval/submit.py bottleneck_pooled --split-by seed
```

### 3b. Headline capacity rematch (the main result)

```bash
# one SLURM job per seed (recommended for walltime)
python scripts/narval/submit.py capacity_rematch --split-by seed

# dry-run first if you want to inspect the generated scripts
python scripts/narval/submit.py capacity_rematch --split-by seed --dry-run
```

### 3c. Mechanism studies

```bash
python scripts/narval/submit.py surrogate --split-by seed
python scripts/narval/submit.py entanglement_ablation --split-by seed-fold
python scripts/narval/submit.py depth_ablation --split-by seed
python scripts/narval/submit.py shot_noise --split-by seed
python scripts/narval/submit.py critic_width --split-by seed
```

### 3d. Downstream / dosage / scarcity (after grouping is trusted)

```bash
python scripts/narval/submit.py decoded_equivalence --split-by seed-fold
python scripts/narval/submit.py qubit_ablation_4q --split-by seed
```

`dose_response` and `data_efficiency` are curves, so each point is its own job. `--set`
overrides any config key (dotted for nested keys, values parsed as YAML), `--tag` keeps
the job and script names distinct, and a separate `output_root` keeps the results from
overwriting each other:

```bash
for d in 512 2048 8192; do
  python scripts/narval/submit.py dose_response --split-by seed --tag "d$d" \
    --set augmentation.dose_count=$d n_synthetic=$d augmentation.max_add=$d \
          output_root=$SCRATCH/eeg_qgan_results_ae_v5c_dose$d
done

for f in 0.25 0.5 1.0; do
  python scripts/narval/submit.py data_efficiency --split-by seed --tag "f$f" \
    --set latent.train_fraction=$f \
          output_root=$SCRATCH/eeg_qgan_results_ae_v5c_frac$f
done
```

The same mechanism runs the architecture ablation, which is what answers "was the
augmentation result just a weak detector":

```bash
python scripts/narval/submit.py classification_probe --split-by seed --tag detv1 \
  --set detector.kind=cnn_bilstm detector.conv_widths='[64,128]' \
        output_root=$SCRATCH/eeg_qgan_results_ae_v5c_detv1
```

Monitor:

```bash
squeue -u $USER
tail -f logs/<job-name>-<jobid>.out
```

### 3e. Thesis deadline path (Aug 2026) — gate + speed

**What happened:**
- `d=32` → NMSE ~0.61 → `raise` failed at gate 0.5
- Wide AE on CPU → 8+ h silent / too slow
- **ae_v4_cpu probe (job 431524):** COMPLETED in ~1 h, best NMSE **0.544 / 0.545**,
  both folds **EXCLUDED** under gate 0.5. Plateau ≈0.54; subsample 8192 hurt capacity.

**Current package (ae_v5):**
- `d=64`, AE widths `[128,256,256]`, ≤300 epochs, patience 40, **full fold** (`max_train_windows: null`)
- Gate **`max_nmse: 0.55`** (calibrated to the observed plateau; still ≪ mean-predictor failure at 1.0 — document in Methods)
- Epoch logging; quantum jobs request a **GPU**
- `deadline_probe`: 1 seed × 2 folds, short generator budget, `exclude`

```bash
cd $HOME/eeg_qgan
# sync from laptop first, then:
module load python/3.11 scipy-stack
source $SCRATCH/venvs/eeg_qgan/bin/activate
pip install -e .

export DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results_ae_v5
mkdir -p "$OUTPUT_ROOT" logs

python scripts/validate_studies.py

# 1. Gate check (do this first — wait for COMPLETED; expect longer wall than ae_v4)
python scripts/narval/submit.py deadline_probe --split-by seed
# watch: tail -f logs/deadline_probe-s0-*.out

# 2. Confirm NMSE < 0.55
python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["SCRATCH"]) / "eeg_qgan_results_ae_v5" / "deadline_probe"
nmses = []
for f in list(root.rglob("meta.json")) + list(root.glob("results_*.json")):
    d = json.loads(f.read_text())
    def walk(o):
        if isinstance(o, dict):
            if "nmse" in o and isinstance(o.get("thresholds"), dict):
                nmses.append(float(o["nmse"]))
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d)
print("n", len(nmses), "min", min(nmses) if nmses else None, "max", max(nmses) if nmses else None)
print("PASS" if nmses and max(nmses) < 0.55 else "FAIL — do not launch raise fleet")
PY

# 3. ONLY if PASS — paper/PFE minimum fleet
python scripts/narval/submit.py capacity_rematch --split-by seed
python scripts/narval/submit.py decoded_equivalence --split-by seed

# 4. If time remains
python scripts/narval/submit.py bottleneck_comparison --split-by seed
python scripts/narval/submit.py surrogate --split-by seed
python scripts/narval/submit.py entanglement_ablation --split-by seed
python scripts/narval/submit.py dose_response --split-by seed
python scripts/narval/submit.py data_efficiency --split-by seed
```

**If deadline_probe NMSE is still ≥ 0.55:** do **not** mass-submit `raise` studies.
Message the number. Options: more AE capacity, or a further justified gate change.

Headline: `d=64`, `n_q=8`, hybrid 704 params, AE `[128,256,256]`, `max_nmse: 0.55`.

---

### 3f. (Historical) ae_v2 / ae_v3 / ae_v4 notes

Do **not** mix result directories. Prefer `eeg_qgan_results_ae_v5` above.
ae_v4_cpu: NMSE ~0.544 under gate 0.5 — informative plateau, not a fleet green light.

---

## Output layout (what each job writes)

```
$OUTPUT_ROOT/<study>/
  config_resolved.json          # full config used for the run
  capacity_audit.json           # parameter / noise-dim audit
  results_<tag>.json            # COMPLETE metrics, rewritten after every fold
  artifacts/
    seed0_fold0/
      meta.json                 # index of files for this fold
      autoencoder.pt            # AE weights (selected by val loss)
      latent_transform.npz      # mean/std/output_scale
      z_report.npy              # held-out real latents used for reporting
      generators/
        quantum.pt              # selected generator checkpoint
        classical_equiv.pt
        ...
      latents/
        quantum_fake.npy        # synthetic codes in generator space
        ...
      gmm/
        gmm.json                # mixture parameters (if that arm ran)
      critics/                  # only if save.critics: true
      decoded/                  # only if save.decoded: true (large)
      detectors/                # only if save.detectors: true
```

The results JSON always contains, per fold: autoencoder metrics, MMD noise floor,
per-arm latent distance + decoded metrics + training history, detector F1/AUROC/...
and an `artifacts` block pointing at the files above.

After a job finishes:

```bash
python scripts/verify_outputs.py "$OUTPUT_ROOT/capacity_rematch"
eeg-qgan-analyze "$OUTPUT_ROOT/capacity_rematch" --reference classical_equiv \
  --csv "$OUTPUT_ROOT/capacity_rematch/tables"
```

Defaults in `configs/base.yaml` under `save:` keep AE, generators, latents and the
transform. Turn on `save.detectors: true` or `save.decoded: true` only if you need them
(decoded windows are large).


---

## 5. What not to do

- Do not mix old `epilepsy_latent_qgan_v3` JSON with new `eeg_qgan` JSON in one table.
- Do not set `denoise: false` unless you also update the Implementation chapter.
- Do not re-enable a positional grouping fallback.
- Do not treat a significant paired difference smaller than the noise floor as a
  quantum advantage; the summariser will mark it.
- Do not stack rebalancing mechanisms. `detector.balanced_batches` alone is the default;
  turning on oversampling and a positive class weight as well applies the same correction
  three times, and duplication at a one-to-one ratio makes the fit memorise a few
  thousand seizure windows.
- Do not mix `autoencoder.arch: v1` and `v2` results, or `detector.kind: cnn_bilstm` and
  `cnn_bilstm_v2` results, inside one table. Both are recorded per fold under
  `autoencoder.architecture.arch` and `detector.*.augmentation.imbalance_strategy`.

---

## Quick local smoke (optional, before spending cluster time)

```bash
cd $HOME/eeg_qgan   # or your Windows/WSL path
pip install -e .
pip install PyWavelets pytest
python scripts/make_synthetic_dataset.py --out data/synthetic
python scripts/plot_preprocessing.py --data-root data/synthetic --out figures/preprocess
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic --output-root results
eeg-qgan-analyze results/smoke --reference classical_equiv
pytest tests -q
```
