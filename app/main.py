# app/main.py
import uuid
import tempfile
import os
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, status

from app.services.ingestion import process_pdf
from app.api.query import router as query_router

app = FastAPI(
    title="RAG API",
    description="PDF ingestion and semantic retrieval",
    version="0.2.0",
)

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(query_router, prefix="/api", tags=["retrieval"])

# ── Ingestion endpoints ────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def read_root():
    return {"message": "system online — ready for API requests"}


@app.post(
    "/uploadfiles/",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
    summary="Upload a PDF for processing",
)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="A single PDF file")],
):
    task_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    background_tasks.add_task(process_pdf, tmp_path, task_id)

    return {"message": "processing started", "task_id": task_id}