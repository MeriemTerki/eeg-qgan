"""End-to-end verification of the deployed demo against the documented claims."""
import json
import urllib.request as u

B = "http://127.0.0.1:8000"


def get(path):
    return json.load(u.urlopen(B + path, timeout=120))


def post(path, body):
    req = u.Request(
        B + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(u.urlopen(req, timeout=300))


ok = []
bad = []


def check(name, got, want, tol=None):
    if tol is not None:
        good = got is not None and abs(float(got) - float(want)) <= tol
    else:
        good = got == want
    (ok if good else bad).append(f"{name}: got {got!r} want {want!r}")


print("=== legacy endpoints still work ===")
h = get("/api/health")
print(" health:", json.dumps(h))
check("weight_source", h["weight_source"], "campaign")
check("data_source", h["data_source"], "tuh-sz-holdout")
s = get("/api/samples")
print(f" samples: {len(s['samples'])} -> {s['samples'][0]['id']} .. {s['samples'][-1]['id']}")
rec = post("/api/reconstruct", {"sample_id": "tuh_ictal_01"})
print(f" reconstruct nmse={rec['nmse']:.4f}")
cmp_old = post("/api/compare", {"sample_id": "tuh_ictal_01", "seed": 42})
print(f" compare arms: {list(cmp_old['generated'])}")

print("\n=== pipeline overview ===")
o = get("/api/pipeline")
check("live_detection", o["available"]["live_detection"], True)
check("n_real_windows", o["available"]["n_real_holdout_windows"], 48)
check("n_real_clips", o["available"]["n_real_holdout_clips"], 6)
check("clip seconds", o["clip"]["seconds"], 10.24, 0.001)
check("windows/synth clip", o["clip"]["windows_per_synthetic_clip"], 9)
check("n detectors", len(o["detectors"]), 4)
check("campaign fold", o["campaign"]["fold_name"], "dev")
check("campaign ae_nmse", o["campaign"]["ae_nmse"], 0.389, 0.002)
check("ictal mean nmse", o["ictal_reconstruction"]["mean_nmse"], 0.483, 0.005)
print(" detector thresholds:")
for d in sorted(o["detectors"], key=lambda x: x["name"]):
    print(f"   {d['label']:<32} thr={d['threshold']:.3f} params={d['n_params']:,}")
    check(f"{d['name']} params", d["n_params"], 1645622)

print("\n=== comparison: three levels ===")
c = get("/api/pipeline/comparison")
check("baseline auroc", c["baseline"]["auroc"], 0.7897, 0.001)
check("baseline n_test", c["baseline"]["n_test"], 27324)
rows = {r["arm"]: r for r in c["rows"]}
print(f"   {'arm':<24}{'L1 live':>10}{'L1 rec':>10}{'sW1':>9}{'ACF':>9}{'AUROC':>9}{'dAUROC':>9}")
for arm in ("quantum", "classical_equiv", "gmm"):
    r = rows[arm]
    l1, l2, l3 = r["level1"], r["level2"], r["level3"]
    print(
        f"   {r['label']:<24}{l1['mmd2']:>10.5f}{l1['recorded_mmd2']:>10.5f}"
        f"{l2['sliced_wasserstein']:>9.4f}{l2['autocorrelation_distance']:>9.4f}"
        f"{l3['auroc']:>9.4f}{l3['delta_auroc']:>+9.4f}"
    )
    # live Level 1 must land close to the recorded value
    check(f"{arm} L1 live~rec", l1["mmd2"], l1["recorded_mmd2"], 0.005)

# documented orderings
check("L1: hybrid < twin", rows["quantum"]["level1"]["mmd2"] < rows["classical_equiv"]["level1"]["mmd2"], True)
check("L1: gmm closest", rows["gmm"]["level1"]["mmd2"] < rows["quantum"]["level1"]["mmd2"], True)
check(
    "L2: hybrid wins sW1",
    rows["quantum"]["level2"]["sliced_wasserstein"] < rows["classical_equiv"]["level2"]["sliced_wasserstein"],
    True,
)
check(
    "L2: twin wins ACF",
    rows["classical_equiv"]["level2"]["autocorrelation_distance"]
    < rows["quantum"]["level2"]["autocorrelation_distance"],
    True,
)
check("thesis baseline", c["thesis"]["baseline_auroc"], 0.821, 0.0001)
check("thesis hybrid delta", c["thesis"]["arms"]["quantum"]["delta_auroc"], 0.0004, 1e-9)
check("thesis hybrid interval", c["thesis"]["arms"]["quantum"]["interval_excludes_zero"], False)

print("\n=== pipeline run per arm ===")
for arm in ("quantum", "classical_equiv", "gmm"):
    r = post("/api/pipeline/run", {"arm": arm, "sample_id": "tuh_ictal_01", "seed": 42, "n_codes": 256})
    s3 = r["stage3_generation"]
    s4 = r["stage4_detection"]
    base = s4["real_clips"]["scores"]["baseline"]
    syn = s4["synthetic_clip"]["scores"]
    own = syn.get(f"augmented_{arm}")
    print(
        f"   {r['label']:<24} gen={s3['generator_latency_ms']:>7.1f}ms "
        f"L1={r['level1_latent']['mmd2']:.5f} "
        f"real_fire={base['n_fired']}/{base['n_clips']} "
        f"syn_own_p={own['probabilities'][0]:.3f} syn_base_p={syn['baseline']['probabilities'][0]:.3f}"
    )
    check(f"{arm} has 4 stages", all(
        k in r for k in ("stage1_data", "stage2_representation", "stage3_generation", "stage4_detection")
    ), True)
    check(f"{arm} has levels 1-2", all(k in r for k in ("level1_latent", "level2_decoded")), True)
    check(f"{arm} gate passed (fold)", r["stage2_representation"]["gate"]["passed"], True)
    check(f"{arm} real clips scored", base["n_clips"], 6)

# the documented claim: baseline fires on 5 of 6 real ictal clips
r = post("/api/pipeline/run", {"arm": "quantum", "sample_id": "tuh_ictal_01", "seed": 42, "n_codes": 256})
check("baseline fires 5/6", r["stage4_detection"]["real_clips"]["scores"]["baseline"]["n_fired"], 5)

print("\n" + "=" * 62)
print(f"PASS {len(ok)}   FAIL {len(bad)}")
for b in bad:
    print("  FAIL:", b)
if not bad:
    print("Every documented claim reproduced.")
