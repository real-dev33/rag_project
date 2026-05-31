import uuid
import tempfile
import os
from typing import Annotated 
from fastapi import FastAPI,File,UploadFile,BackgroundTasks,status # type: ignore
from app.services.ingestion import process_pdf # type: ignore
app = FastAPI()
@app.get("/")
def read_root():
    return {"message":"system online ready for API requests"}
@app.post("/uploadfiles/",status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="A single file")]
):
    # Generate unique ID
    task_id = str(uuid.uuid4())

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Queue the real background task
    background_tasks.add_task(process_pdf, tmp_path, task_id)

    return {"message": "processing started", "taskid": task_id}
