# Demo walkthrough for the defense

About six minutes. Start the server before the session, not in front of the jury.

```powershell
cd C:\Users\meriemterki\Documents\Thesis\eeg_qgan
python demo/server.py
```

Open http://127.0.0.1:8000 and leave it on the **Full pipeline** tab. Have http://127.0.0.1:8000/docs open in a second browser tab.

---

## 0. Before you click anything (30 s)

Point at the amber scope banner.

> "This is a deployment of the architecture, running the saved campaign checkpoints. It serves one fold: seed 42, official holdout `dev`. The manuscript headline pools four seed-by-fold records, so the detection column here is one fold and is wider than the pooled mean. I will come back to that."

Say the disclaimer once: research prototype, not a medical device.

---

## 1. The stages (1 min)

Point at the four stage cards.

> "One pipeline, four stages. Data and preprocessing. Representation through the frozen autoencoder. Generation. Detection on a ten-second clip. Everything except the generator body is shared, so if two arms differ, the difference belongs to the generator."

---

## 2. Run the hybrid (1 min 30 s)

Arm: **Hybrid Style-qGAN**. Seed 42. Codes 256. Click **Run whole pipeline**.

While it runs:

> "The generator is a PennyLane circuit on CPU, eight qubits, four re-uploading layers, a ring of controlled-Z gates, and analytic Pauli-Z readout. Seven hundred and four parameters, and eighty-two percent of that is the classical readout."

When the cards fill:

> "Stage 2, the frozen host: this window reconstructs at about `0.56` normalised error. The gate was `0.55` on the fold's mixed validation split, where this fold recorded `0.389` and passed. These cached windows are all ictal, and the ictal class carries more amplitude, so it reconstructs worse than that average. The host is weakest on exactly the class we care about, and that matters later."

> "Stage 3: two hundred and fifty-six codes in about a quarter of a second, then the frozen decoder. Level 1, computed now: MMD² of about `0.015` against the real holdout codes, and the campaign recorded `0.016`."

---

## 3. The detector panel (1 min)

Point at the right-hand column.

> "Stage 4. Eight windows of `1.28` seconds pack into one clip of `10.24` seconds, which is the length at which a clinician recognises an event. All four detectors are frozen, identical in architecture, one and a half million parameters, and each carries the operating point chosen on its own real validation clips."

> "On the six real holdout ictal clips the baseline fires on five of six. Then one synthetic clip from the hybrid: the hybrid-augmented detector scores it near one, and the real-only baseline scores it near zero. The detector trained with hybrid synthetics recognises hybrid synthetics. The baseline does not. That is a statement about the synthetic distribution, not about clinical benefit."

---

## 4. Compare all arms (1 min 30 s)

Click **Compare all arms**.

> "Three levels, every arm, one table. Levels 1 and 2 are recomputed now. Level 3 needs the corpus, which is on the cluster, so it is read from the campaign record and labelled as recorded."

> "Level 1: the hybrid at `0.015` against the twin at `0.096`, same critic, `704` against `709` parameters. About six times closer. The Gaussian mixture, which uses no critic at all, is closer than both at `0.001`. So the closest model on this metric is not adversarial."

> "Level 2 is where the transfer starts to fail, and it is not uniform. The hybrid is closer on sliced Wasserstein. The twin is closer on autocorrelation, `0.037` against `0.067`. The hybrid buys distributional closeness at the cost of temporal structure, and temporal structure is what a spectrogram detector reads."

Then the honest close, pointing at the reading line under the table:

> "Level 3 on this one fold gives the hybrid plus `0.014`. Pooled over the four records in the manuscript it is plus `0.0004` with an interval that contains zero, while the twin and the mixture both lift. One fold is not the campaign. That is why the thesis reports intervals over four records, and why the conclusion is that a latent win is not a detection win at full dose."

---

## 5. The API tab (30 s)

Switch to **API audit**, then to `/docs`.

> "The website is an API client. Every figure you just saw came from an HTTP call, and this is the last request and response. The OpenAPI page lists the endpoints: `/api/pipeline/run` takes an arm, a sample, a seed, and a code count, and returns a result at every stage."

---

## If something goes wrong

| Symptom | Do this |
|---|---|
| Port already in use | A server is already running. Just open the browser. |
| Status card says `architecture-only` | Say it plainly: the circuit is the thesis circuit, the weights are not the campaign. Then use the recorded columns. |
| First hybrid run is slow | That delay is the circuit. Say so and keep talking. |
| A juror asks for a live AUROC | The cached windows are ictal only, so one class cannot form an AUROC. The recorded value used `27,324` real holdout clips with both classes. |

## Questions you should expect

**"Is this running the real models?"**
Yes. The autoencoder, the three generators, and the four detectors are the campaign checkpoints from `thesis_figure_capture/artifacts/seed0_fold0`. Levels 1 and 2 are recomputed live and land within about `0.001` of the recorded values.

**"Why is the detection number here different from your thesis?"**
Because this is one fold and the thesis pools four. The demo deploys the architecture; it does not restate the evaluation.

**"Could this run in a hospital?"**
No, and it is not proposed for that. It is a research prototype on a public corpus, one detector, one fold, with no prospective validation and no regulatory work.

**"Is the quantum part doing the work?"**
Eighty-two percent of the hybrid budget is the classical readout, the product circuit without entanglement scores better, and a classical Fourier model of the matching class is closer still. The circuit is a compact generator under a named contract, not a demonstrated quantum advantage.
