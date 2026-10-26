#!/usr/bin/env python3
"""TUH-SZ research workstation: FastAPI + clinical viewer."""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.inventory import inventory  # noqa: E402
from demo.paths import STATIC_DIR  # noqa: E402
from demo.service import get_engine, select_study, window_payload  # noqa: E402
from demo.paths import discover_studies  # noqa: E402

app = FastAPI(
    title="TUH-SZ seizure EEG research workstation",
    description=(
        "Research APIs for the Engineering thesis: hybrid latent Style-qGAN, "
        "capacity-matched twin, and Gaussian mixture on TUH-SZ 22 x 256 windows. "
        "Not a medical device."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateBody(BaseModel):
    arm: str = "quantum"
    seed: int | None = Field(default=42)
    n: int = Field(default=1, ge=1, le=4)


class ReconstructBody(BaseModel):
    sample_id: str


class CompareBody(BaseModel):
    sample_id: str
    seed: int = 42


class PipelineBody(BaseModel):
    arm: str = Field(default="quantum", description="Generator arm to run end to end")
    sample_id: str | None = Field(default=None, description="Holdout window; first if omitted")
    seed: int = Field(default=42)
    n_codes: int = Field(
        default=256,
        ge=9,
        le=2048,
        description="Codes to generate for the live Level 1 reading",
    )


@app.get("/api/health")
def health() -> dict:
    engine = get_engine()
    return {
        "ok": True,
        "weight_source": engine.weight_source,
        "data_source": engine.data_source,
        "n_samples": len(engine.catalog),
        "n_models": len(engine.loaded),
    }


@app.get("/api/meta")
def meta() -> dict:
    return get_engine().meta()


@app.get("/api/architecture")
def architecture() -> dict:
    return get_engine().architecture()


@app.get("/api/inventory")
def inventory_api() -> dict:
    return inventory()


@app.get("/api/samples")
def samples() -> dict:
    return {
        "data_source": get_engine().data_source,
        "samples": get_engine().list_samples(),
    }


class StudyBody(BaseModel):
    study_id: str = Field(..., description="Known study id, e.g. entanglement_deadline")


@app.get("/api/studies")
def studies() -> dict:
    engine = get_engine()
    return {
        "active_study": engine.study_id,
        "active_fold": str(engine.fold_dir) if engine.fold_dir else None,
        "loaded_arms": list(engine.loaded),
        "studies": discover_studies(),
        "sync_hint": (
            "Clinical fold only. From PowerShell: .\\demo\\sync_from_narval.ps1 "
            "(interactive Narval MFA if re-download is needed)."
        ),
    }


@app.post("/api/studies/select")
def studies_select(body: StudyBody) -> dict:
    try:
        engine = select_study(body.study_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown study '{body.study_id}'") from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "ok": True,
        "active_study": engine.study_id,
        "loaded_arms": list(engine.loaded),
        "fold": str(engine.fold_dir) if engine.fold_dir else None,
        "notes": engine.notes[-8:],
        "arm_status": engine.arm_status(),
    }


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str) -> dict:
    engine = get_engine()
    try:
        item, window = engine.load_sample(sample_id)
        detection = engine.detect_holdout(sample_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown recording '{sample_id}'") from exc
    return {
        "sample": item,
        "window": window_payload(window),
        "detection": detection,
    }


@app.get("/api/samples/{sample_id}/detect")
def sample_detect(sample_id: str) -> dict:
    try:
        return get_engine().detect_holdout(sample_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown recording '{sample_id}'") from exc


@app.post("/api/generate")
def generate(body: GenerateBody) -> dict:
    try:
        return get_engine().generate(body.arm, seed=body.seed, n=body.n)
    except KeyError as exc:
        raise HTTPException(400, f"Unknown model '{body.arm}'") from exc


@app.post("/api/reconstruct")
def reconstruct(body: ReconstructBody) -> dict:
    try:
        return get_engine().reconstruct_sample(body.sample_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown recording '{body.sample_id}'") from exc


@app.post("/api/compare")
def compare(body: CompareBody) -> dict:
    try:
        return get_engine().compare(body.sample_id, seed=body.seed)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown recording '{body.sample_id}'") from exc


@app.get("/api/pipeline")
def pipeline_overview() -> dict:
    """Stages, evidence levels, detectors, and what is live versus quoted."""
    return get_engine().pipeline_overview()


@app.post("/api/pipeline/run")
def pipeline_run(body: PipelineBody) -> dict:
    """Run one generator arm through all four stages and report every level."""
    try:
        return get_engine().run_pipeline(
            body.arm, sample_id=body.sample_id, seed=body.seed, n_codes=body.n_codes
        )
    except KeyError as exc:
        raise HTTPException(400, f"Unknown model '{body.arm}'") from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/pipeline/live-compare")
def pipeline_live_compare(body: PipelineBody) -> dict:
    """Regenerate from every saved model and score generation plus detection live."""
    try:
        return get_engine().live_compare(
            sample_id=body.sample_id, seed=body.seed, n_codes=body.n_codes
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/pipeline/comparison")
def pipeline_comparison() -> dict:
    """Every arm at three levels, side by side."""
    return get_engine().comparison()


@app.get("/api/detectors")
def detectors() -> dict:
    engine = get_engine()
    return {
        "detectors": engine.detectors.describe(),
        "notes": engine.detectors.notes,
        "recorded": engine.campaign.detection() if engine.campaign.available else {},
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
