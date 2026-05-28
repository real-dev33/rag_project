from fastapi import APIRouter, UploadFile, BackgroundTasks
from services.ingestion import process_pdf
import uuid, tempfile, os

router = APIRouter()

@router.post("/upload", status_code=202)
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Add background task to process the file
    background_tasks.add_task(process_pdf, tmp_path, task_id)

    return {"task_id": task_id, "message": "Processing started"}