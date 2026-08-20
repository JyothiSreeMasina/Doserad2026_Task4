"""Inference server implementing Grand Challenge's `invoke` HTTP API.

Loads the model once at startup (not per-invoke -- see the module docstring
in process.py: each invoke has a timeout, and loading weights inside it
would eat into the per-beam time budget), then serves /health and /invoke.
"""
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Response, status
from uvicorn.config import LOGGING_CONFIG

import process

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model, cfg = process.init_model(device)
    STATE["model"] = model
    STATE["cfg"] = cfg
    STATE["device"] = device
    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if "model" in STATE:
        return Response(status_code=status.HTTP_200_OK)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


@app.post("/invoke")
async def invoke():
    process.run(STATE["model"], STATE["cfg"], STATE["device"])
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
