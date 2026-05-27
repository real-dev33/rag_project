import uuid
from typing import Annotated 
from fastapi import FastAPI,File,UploadFile,BackgroundTasks,status # type: ignore
app = FastAPI()
def process_pdf_background(task_id: str,filename: str):
    print(f"processing{task_id}")
@app.get("/")
def read_root():
    return {"message":"system online ready for API requests"}
@app.post("/uploadfiles/",status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
 background_tasks: BackgroundTasks,file: Annotated[UploadFile,File(description="A single file")]
):
#generate unique id
 task_id= str(uuid.uuid4())
#queue the bg task passing the task_id and file name
 background_tasks.add_task(process_pdf_background,task_id,file.filename)
#return 202 accepted msg

 return {"message":"processing started","taskid":task_id}
