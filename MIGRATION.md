# Migration map

Where each old repository, config and job script went.

## Why there were eleven repositories

The eleven directories were not eleven codebases. Comparing file hashes across them showed
that `epilepsy_latent_qgan_v3` was the newest version of essentially every shared module,
and the other latent-qGAN directories were copies of it that differed only in
configuration: a qubit count, a data fraction, whether the quantum arm was replaced by a
classical one. `epilepsy_qcgan` was different in kind, holding the data loader and the
downstream detector.

Duplicating the source to change a hyperparameter meant that every later fix had to be
applied eleven times, and in practice was not. Here those differences are study overlays,
so a fix lands once.

| Old repository | What actually differed | Replacement |
| --- | --- | --- |
| `epilepsy_latent_qgan` | oldest copy, superseded | history only |
| `epilepsy_latent_qgan_v3` | the live codebase | ported into `eeg_qgan/` |
| `epilepsy_latent_qgan_4q` | `n_qubits: 4` | `configs/studies/qubit_ablation_4q.yaml` |
| `epilepsy_latent_qgan_4q_v3` | `n_qubits: 4`, newer base | `configs/studies/qubit_ablation_4q.yaml` |
| `epilepsy_latent_qgan_vqc_4q` | plain VQC ansatz, 4 qubits | `qubit_ablation_4q.yaml` with `ansatz: vqc` |
| `epilepsy_latent_qgan_vqc_8q` | plain VQC ansatz, 8 qubits | `capacity_rematch.yaml` with `ansatz: vqc` |
| `epilepsy_latent_qgan_gan` | classical arm only | `capacity_rematch.yaml`, classical arms |
| `epilepsy_latent_qgan_dataeff_25` | 25% of training data | `configs/studies/data_efficiency.yaml` |
| `epilepsy_latent_qgan_dataeff_50` | 50% of training data | `configs/studies/data_efficiency.yaml` |
| `epilepsy_latent_qgan_dataeff_100` | 100% of training data | `configs/studies/data_efficiency.yaml` |
| `epilepsy_qcgan` | data loader and detector | `eeg_qgan/data/tuh.py`, `eeg_qgan/models/detector.py` |

The three `dataeff` repositories become one study because the data fraction is a swept
axis, not a separate program. Running the sweep in one job also means all fractions share
an autoencoder and a fold partition, which is what makes the comparison across fractions
meaningful rather than confounded with the split.

## Configs

`epilepsy_latent_qgan_v3` carried 28 config files whose names encode what is now a command
line flag. `experiment_fair_equiv_8q_cpu_s1.yaml` differs from
`experiment_fair_equiv_8q_cpu.yaml` only in which seed it runs, and
`experiment_decoded_equiv_cpu_f23.yaml` only in which folds.

| Old configs | Replacement |
| --- | --- |
| `experiment_fair_equiv_8q*.yaml` (5 files) | `capacity_rematch.yaml` + `--split-by seed` |
| `experiment_fair_equiv_4q*.yaml` (4 files) | `qubit_ablation_4q.yaml` + `--split-by seed` |
| `experiment_fair_v2/v3`, `experiment_fair_three_arm*` | `capacity_rematch.yaml` (all arms in one run) |
| `experiment_decoded_equiv*.yaml` (5 files) | `decoded_equivalence.yaml` + `--split-by seed-fold` |
| `experiment_phase1_floor*.yaml` | the MMD noise floor, now computed in every fold |
| `experiment_phase1_unified_eval*.yaml` (4 files) | default: every arm is evaluated in both spaces |
| `experiment_entanglement_ablation.yaml` | `entanglement_ablation.yaml` |
| `experiment_depth_ablation_d2/d4.yaml` | `depth_ablation.yaml` (both depths, one study) |
| `experiment_phase0_product_pt.yaml` | `surrogate.yaml`, product topology arm |
| `experiment_gan_compare.yaml`, `experiment_fair_gan.yaml` | `capacity_rematch.yaml` classical arms |
| `experiment_latent_clf*.yaml` | **not carried over**, see below |

The noise floor deserves a note. It used to be a separate study, which meant most results
were read without it. It is now computed inside every fold and stored on every row,
because a distance is uninterpretable without the real-against-real reference at the same
sample size and the same estimator.

## Job scripts

31 shell scripts in `v3` and 33 in `epilepsy_qcgan` collapse into one generator. The
`_s0`/`_s1`/`_s2` and `_f01`/`_f23`/`_f4` suffixes were manual work splitting, which is
now a flag.

| Old scripts | Replacement |
| --- | --- |
| `narval_fair_equiv_8q_cpu_s{0,1}.sh` | `submit.py capacity_rematch --split-by seed` |
| `narval_fair_equiv_4q_cpu_s{0,1,2}.sh` | `submit.py qubit_ablation_4q --split-by seed` |
| `narval_decoded_equiv_cpu_f{01,23,4}.sh` | `submit.py decoded_equivalence --split-by seed-fold` |
| `narval_phase1_unified_cpu_f{01,23,4}.sh`, `_a`, `_b` | `submit.py capacity_rematch --split-by seed-fold` |
| `narval_phase1_floor*.sh` | none needed: the floor is computed per fold |
| `final_e1/e2/e3.sh`, `final_tier0.sh`, `narval_p2_remaining.sh` | `submit.py <study> --split-by seed` |
| `export_v3_report_rows.sh`, `run_h2_export.sh`, `narval_recompute_h2.sh` | `eeg-qgan-analyze` |
| `spine_mechanism.sh` | `submit.py surrogate` and `submit.py bottleneck_comparison` |

Resource requests are no longer copied between files. `submit.py` derives walltime, cores,
memory and whether to ask for a GPU from the study itself, and asks for CPU only when the
study is simulator-bound, which both queues faster and stops a GPU idling while PennyLane
runs on the processor.

## Studies

| Study | Question it answers |
| --- | --- |
| `capacity_rematch` | the headline comparison, quantum against matched classical arms |
| `qubit_ablation_4q` | does the conclusion hold at 4 qubits |
| `entanglement_ablation` | does entanglement topology matter once the metric is corrected |
| `surrogate` | can a classical Fourier basis reproduce the quantum arm's behaviour |
| `depth_ablation` | does circuit depth change anything |
| `shot_noise` | how much of the result is finite-shot noise |
| `bottleneck_comparison` / `bottleneck_pooled` | how much the autoencoder bottleneck fix moves the numbers |
| `critic_width` | is the result a property of the generator or of the critic |
| `dose_response` | does downstream benefit depend on augmentation dose |
| `data_efficiency` | does any advantage appear only when data is scarce |
| `decoded_equivalence` | do latent differences survive decoding to signal space |
| `smoke` | a fast end-to-end check on synthetic data |

## What is not carried over

### The quantum classifier head

`experiment_latent_clf.yaml` and `experiment_latent_clf_v2.yaml` put a variational circuit
in the *classifier* rather than the generator, through `LatentHybridVQC` and
`QuantumClassifierHead`. The same axis appears in `epilepsy_qcgan` as the `*_vqc`
classifiers reached through `build_classifier`.

That is a different research question from the one this package addresses. Everything here
concerns quantum *generative* modelling, evaluated with a fixed classical detector so that
changes in downstream metrics can be attributed to the synthetic data. Putting a circuit in
the detector as well would confound the two. The detector is therefore classical only:
`detector.kind` accepts `cnn_bilstm` and `latent_mlp`.

If the thesis needs that axis, it is an additive change rather than a rewrite: port
`QuantumClassifierHead` into `eeg_qgan/models/detector.py` as a third `kind`, and give it a
study of its own so the generative comparison stays interpretable. Until then the old
repositories remain the reference for those two studies.

### Old results

Results in the old repositories are left where they are. They were produced under the
grouping and selection procedures the audit criticised, so they are kept as a record but
should not be pooled with new output. `AUDIT_FIXES.md` says which of them are expected to
change and in which direction.

### Superseded modules

The inventory of `epilepsy_latent_qgan_v3` confirmed that five modules were dead: an older
adversarial path (`train_latent_gan.py`) and everything only it used, namely `train_vae.py`,
`multichannel_vae.py`, `pure_quantum_generator.py`, and `latent_discriminator.py`. They are
not ported. The live critic used by `train_latent_adv` is now `eeg_qgan/models/critic.py`.
The legacy single-channel loader (`eeg_dataset.py`, used by the v1 to v8 studies, which
flattened all channels into one) is also not ported, since every current study is
multichannel.
