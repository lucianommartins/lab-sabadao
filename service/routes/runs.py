# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import os
import uuid
import json
import asyncio
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
from googleapiclient import discovery
from google.auth import default
from service.models.schemas import RunTriggerRequest
from service.utils.storage import list_runs, get_run_details, upload_run_folder_to_gcs, is_gbench_process_running
from service.config import settings

logger = logging.getLogger(__name__)


router = APIRouter()

# Memory state for active runs
ACTIVE_RUNS = {}

@router.get("", response_model=list[dict])
def list_all_runs():
    completed = list_runs()
    active = []
    active_ids = set()
    for rid, info in ACTIVE_RUNS.items():
        if info["process"].poll() is not None:
            continue
        active_ids.add(rid)
        active.append({
            "run_id": rid,
            "status": info["status"],
            "created_at": info["created_at"],
            "completed_at": None,
            "tags": info.get("tags", []),
            "results": {
                "quality": {
                    "raw_results": {
                        "counts": {
                            "total": info.get("total_scenarios", 0),
                            "passed": 0,
                            "failed": 0
                        }
                    }
                }
            }
        })
    # Filter out active runs from completed to avoid duplicates
    completed_filtered = [r for r in completed if r.get("run_id") not in active_ids]
    return active + completed_filtered

@router.get("/{run_id}")
def get_run(run_id: str):
    if run_id in ACTIVE_RUNS:
        info = ACTIVE_RUNS[run_id]
        if info["process"].poll() is None:
            return {
                "run_id": run_id,
                "status": info["status"],
                "created_at": info["created_at"],
                "completed_at": None,
                "tags": info.get("tags", []),
                "results": {
                    "quality": {
                        "raw_results": {
                            "counts": {
                                "total": info.get("total_scenarios", 0),
                                "passed": 0,
                                "failed": 0
                            }
                        }
                    }
                }
            }
    details = get_run_details(run_id)
    if not details:
        raise HTTPException(status_code=404, detail="Run not found")
    return details

@router.post("", response_model=dict)
def trigger_run(request: RunTriggerRequest, background_tasks: BackgroundTasks):
    for rid, info in ACTIVE_RUNS.items():
        if info["process"].poll() is None:
            raise HTTPException(status_code=409, detail=f"A benchmark run ({rid}) is already in progress.")
            
    # Check filesystem for active runs to prevent concurrent executions across server restarts
    if settings.is_local and settings.LOCAL_RESULTS_DIR.exists():
        for run_dir in settings.LOCAL_RESULTS_DIR.iterdir():
            if run_dir.is_dir():
                active_run_file = run_dir / "active_run.json"
                if active_run_file.exists():
                    try:
                        with open(active_run_file) as f:
                            meta = json.load(f)
                        if meta.get("status") in ["running", "pending"]:
                            pid = meta.get("pid", 0)
                            if is_gbench_process_running(pid):
                                raise HTTPException(
                                    status_code=409, 
                                    detail=f"A benchmark run ({meta.get('run_id')}) is already active in the OS (PID: {pid})."
                                )
                    except HTTPException:
                        raise
                    except Exception:
                        pass
            
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow().isoformat() + "Z"
    
    background_tasks.add_task(_execute_benchmark, run_id, request)
    
    return {"run_id": run_id, "status": "running", "message": "Run triggered successfully."}

def _execute_benchmark(run_id: str, req: RunTriggerRequest):
    cmd = [sys.executable, "-u", "-m", "gbench.cli", "--models", req.model_id, "--format", req.format]
    
    if req.run_type == "performance":
        cmd.append("--no-quality")
    elif req.run_type == "quality":
        cmd.append("--quality-only")
        
    if req.preset == "quick":
        cmd.append("--preset=quick")
        
    if settings.is_local:
        local_endpoint = os.getenv("GBENCH_LOCAL_REMOTE_ENDPOINT", "http://localhost:11434/v1")
        cmd.extend(["--remote-endpoint", local_endpoint])
    else:
        remote_url = None
        try:
            credentials, _ = default()
            run_service = discovery.build('run', 'v1', credentials=credentials)
            service_path = f"projects/{settings.GCP_PROJECT}/locations/{settings.GCP_REGION}/services/{settings.VLLM_SERVICE_NAME}"
            response = run_service.projects().locations().services().get(name=service_path).execute()
            remote_url = response['status']['url'] + "/v1"
            logger.info(f"Resolved Cloud Run service URL: {remote_url}")
        except Exception as e:
            logger.error(f"Failed to dynamically resolve Cloud Run service URL: {e}")
            
        env_url = os.getenv("GBENCH_REMOTE_ENDPOINT")
        if env_url:
            remote_url = env_url
            logger.info(f"Using remote endpoint from env: {remote_url}")
            
        if remote_url:
            cmd.extend(["--remote-endpoint", remote_url])
        else:
            logger.error("No remote endpoint configured! Benchmark run might fail.")
        
    cmd.extend(["--results-dir", f"results/{run_id}"])
    
    if req.selected_scenarios:
        cmd.append("--scenarios")
        scenario_ids = [Path(p).stem for p in req.selected_scenarios]
        cmd.extend(scenario_ids)
        
    if req.tags:
        cmd.append("--tags")
        cmd.extend(req.tags)
        
    run_dir = settings.LOCAL_RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = run_dir / "master.log"
    
    with open(master_log_path, "wb", buffering=0) as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(settings.BASE_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT
        )
        
        # Resolve total scenarios count for live progress tracking
        total_scenarios_count = 0
        if req.selected_scenarios:
            total_scenarios_count = len(req.selected_scenarios)
        else:
            try:
                from service.utils.scenarios import build_scenarios_tree
                nodes = build_scenarios_tree()
                def count_files(items):
                    c = 0
                    for item in items:
                        if item.type == "file":
                            c += 1
                        elif item.children:
                            c += count_files(item.children)
                    return c
                total_scenarios_count = count_files(nodes)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to count total scenarios recursively: {e}")
                total_scenarios_count = 10  # Fallback baseline

        ACTIVE_RUNS[run_id] = {
            "status": "running",
            "process": process,
            "log_file": master_log_path,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tags": req.tags or [],
            "total_scenarios": total_scenarios_count
        }
        
        # Write active_run.json
        active_meta = {
            "run_id": run_id,
            "status": "running",
            "pid": process.pid,
            "created_at": ACTIVE_RUNS[run_id]["created_at"],
            "tags": req.tags or [],
            "total_scenarios": total_scenarios_count,
            "model_id": req.model_id
        }
        with open(run_dir / "active_run.json", "w") as f:
            json.dump(active_meta, f, indent=2)
            
        process.wait()
        
        # Update final status on exit
        try:
            active_run_file = run_dir / "active_run.json"
            if active_run_file.exists():
                with open(active_run_file) as f:
                    curr_meta = json.load(f)
                if curr_meta.get("status") == "cancelled":
                    active_meta["status"] = "cancelled"
                else:
                    active_meta["status"] = "completed" if process.returncode == 0 else "failed"
            else:
                active_meta["status"] = "completed" if process.returncode == 0 else "failed"
            with open(active_run_file, "w") as f:
                json.dump(active_meta, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update active_run.json status on exit: {e}")
        
    if not settings.is_local:
        upload_run_folder_to_gcs(run_id, run_dir)

    if run_id in ACTIVE_RUNS:
        del ACTIVE_RUNS[run_id]

@router.post("/{run_id}/cancel")
def cancel_run(run_id: str):
    run_dir = settings.LOCAL_RESULTS_DIR / run_id
    active_run_file = run_dir / "active_run.json"
    
    if run_id in ACTIVE_RUNS:
        info = ACTIVE_RUNS[run_id]
        process = info["process"]
        if process.poll() is None:
            process.terminate()
            
            # Overwrite active_run.json status to cancelled
            if active_run_file.exists():
                try:
                    with open(active_run_file) as f:
                        meta = json.load(f)
                    meta["status"] = "cancelled"
                    with open(active_run_file, "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to update active_run.json status on cancel: {e}")
                    
            del ACTIVE_RUNS[run_id]
            return {"status": "cancelled", "message": "Run aborted successfully."}
        
        del ACTIVE_RUNS[run_id]
        return {"status": "completed", "message": "Run had already finished."}

    # Fallback to filesystem metadata if server restarted/reloaded
    if active_run_file.exists():
        try:
            with open(active_run_file) as f:
                meta = json.load(f)
            
            status = meta.get("status")
            pid = meta.get("pid")
            
            if status == "running" and pid:
                # Check if process is alive in the OS
                try:
                    import os
                    import signal
                    os.kill(pid, 0)
                    
                    logger.info(f"Process {pid} found alive in OS on cancel fallback, sending SIGTERM...")
                    os.kill(pid, signal.SIGTERM)
                    
                    # Update status
                    meta["status"] = "cancelled"
                    with open(active_run_file, "w") as f:
                        json.dump(meta, f, indent=2)
                    return {"status": "cancelled", "message": "Run aborted successfully via fallback."}
                except OSError:
                    # Process is not running
                    meta["status"] = "failed"
                    with open(active_run_file, "w") as f:
                        json.dump(meta, f, indent=2)
                    return {"status": "failed", "message": "Run was not active."}
            else:
                return {"status": status, "message": f"Run is already in {status} status."}
        except Exception as e:
            logger.error(f"Failed to read/cancel run via fallback: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to cancel run: {e}")

    raise HTTPException(status_code=404, detail="Run not found or not active")

@router.get("/{run_id}/stream-logs")
async def stream_logs(run_id: str):
    run_dir = settings.LOCAL_RESULTS_DIR / run_id
    log_path = run_dir / "master.log"
    
    async def log_generator():
        if not log_path.exists():
            for _ in range(10):
                if log_path.exists():
                    break
                await asyncio.sleep(0.5)
            if not log_path.exists():
                yield "data: Error: Log file not found.\n\n"
                return
                
        with open(log_path, "r") as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    is_active = False
                    active_run_file = run_dir / "active_run.json"
                    if active_run_file.exists():
                        try:
                            with open(active_run_file) as af:
                                meta = json.load(af)
                                if meta.get("status") in ["running", "pending"]:
                                    is_active = is_gbench_process_running(meta.get("pid", 0))
                        except Exception:
                            pass
                    if not is_active:
                        is_active = run_id in ACTIVE_RUNS and ACTIVE_RUNS[run_id]["process"].poll() is None
                        
                    if not is_active:
                        yield "data: [PROCESS_COMPLETED]\n\n"
                        break
                    await asyncio.sleep(0.5)
                    
    return StreamingResponse(log_generator(), media_type="text/event-stream")
