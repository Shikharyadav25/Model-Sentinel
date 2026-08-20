"""
FastAPI REST API for ModelSentinel.
Exposes endpoints for model scanning, attack simulation, and system health.
"""

import os
import shutil
import tempfile
from typing import Optional
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.attack.xlsb_attack import EICAR_PAYLOAD, embed_payload, embedding_rate
from src.pipeline.scan import scan_model
from src.utils.weight_io import load_flat_weights, save_flat_weights_as_state_dict

app = FastAPI(
    title="ModelSentinel API",
    description="Steganographic Malware Scanner & X-LSB Attack Simulator for AI Models",
    version="1.0.0",
)

# Enable CORS for local dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Liveness check and system status."""
    calib_exists = os.path.exists("models/calibration.json")
    cnn_exists = os.path.exists("models/fewshot_cnn.pt")
    return {
        "status": "healthy",
        "service": "ModelSentinel",
        "version": "1.0.0",
        "detectors_ready": calib_exists and cnn_exists,
        "calibration_loaded": calib_exists,
        "cnn_checkpoint_loaded": cnn_exists,
    }


@app.get("/models")
def list_sample_models():
    """Lists preloaded benign and sample attacked models in data directory."""
    benign_dir = "data/benign"
    attacked_dir = "data/attacked"
    benign_files = []
    attacked_files = []

    if os.path.exists(benign_dir):
        benign_files = sorted([f for f in os.listdir(benign_dir) if f.endswith((".pt", ".safetensors"))])
    if os.path.exists(attacked_dir):
        attacked_files = sorted([f for f in os.listdir(attacked_dir) if f.endswith((".pt", ".safetensors"))])

    return {
        "benign_models": benign_files,
        "attacked_samples": attacked_files,
    }


@app.post("/scan")
async def scan_uploaded_model(
    file: Optional[UploadFile] = File(None),
    model_path: Optional[str] = Form(None),
):
    """
    Scans an AI model file for steganographic malware.
    Accepts either an uploaded file or a server-side model_path.
    """
    temp_file = None
    try:
        if file is not None:
            ext = os.path.splitext(file.filename)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            target_path = temp_file.name
        elif model_path is not None:
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model path not found: {model_path}")
            target_path = model_path
        else:
            raise HTTPException(status_code=400, detail="Must provide either a file upload or model_path")

        result = scan_model(target_path)
        if file is not None:
            result["filename"] = file.filename

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file is not None and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass


@app.post("/simulate-attack")
async def simulate_attack(
    file: UploadFile = File(...),
    x: int = Form(4),
    payload_type: str = Form("eicar"),
):
    """
    Simulates X-LSB-Attack-Fill steganography on an uploaded model.
    Returns the attacked model file for download.
    """
    if not (1 <= x <= 23):
        raise HTTPException(status_code=400, detail="X must be between 1 and 23.")

    ext = os.path.splitext(file.filename)[1]
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_out.close()

    try:
        shutil.copyfileobj(file.file, temp_in)
        temp_in.close()

        # Select payload
        if payload_type == "random":
            # Generate deterministic pseudo-random bytes
            payload = np.random.bytes(1024)
        else:
            payload = EICAR_PAYLOAD

        # Load, attack, save
        flat = load_flat_weights(temp_in.name)
        attacked_flat = embed_payload(flat, X=x, payload=payload)
        save_flat_weights_as_state_dict(temp_in.name, attacked_flat, temp_out.name)

        out_filename = f"attacked_x{x}_{file.filename}"
        return FileResponse(
            temp_out.name,
            filename=out_filename,
            media_type="application/octet-stream",
        )
    except Exception as e:
        if os.path.exists(temp_out.name):
            os.remove(temp_out.name)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_in.name):
            os.remove(temp_in.name)
