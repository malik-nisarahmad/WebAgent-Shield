from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .agent import load_run, run_agent
from .detector import status as detector_status

load_dotenv()
app = FastAPI(title="WebAgent Shield", version="0.1.0")


class Task(BaseModel):
    task: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"status": "ok", "phase": 3, "detector_status": detector_status()}


@app.post("/v1/agent-runs")
async def create_run(request: Task):
    try:
        return await run_agent(request.task)
    except ValueError as error:
        raise HTTPException(503, str(error)) from error


@app.get("/v1/agent-runs/{run_id}")
def get_run(run_id: str):
    run = load_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run
