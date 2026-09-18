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

import os
import json
import logging
import shutil
import datetime
from pathlib import Path
from typing import Any, Optional
from service.config import settings
from gbench.core.models import registry, MODELS_DIR

logger = logging.getLogger(__name__)

def _get_storage_client():
    # Imported lazily so the service module (and its test collection) does not hard-require the
    # optional google-cloud-storage dependency just to import; it is only needed for GCS operations.
    from google.cloud import storage
    return storage.Client()

def list_staged_models() -> list[str]:
    """List available model directories from local staging or GCS, plus built-in models."""
    models = [m.hf_model_id for m in registry.list_all()]
    
    if settings.is_local:
        if MODELS_DIR.exists():
            for d in MODELS_DIR.iterdir():
                if d.is_dir() and d.name not in models:
                    models.append(d.name)
    else:
        # In prod mode, list GCS bucket prefixes
        try:
            client = _get_storage_client()
            bucket = client.bucket(settings.MODELS_BUCKET)
            iterator = bucket.list_blobs(delimiter="/")
            list(iterator)  # Execute iterator
            for p in iterator.prefixes:
                name = p.rstrip("/")
                if name not in models:
                    models.append(name)
        except Exception as e:
            logger.error(f"Failed to list staged models from GCS: {e}")
            
    return models

def list_runs() -> list[dict]:
    """List all historical benchmark runs (local files or GCS)."""
    runs = []
    if settings.is_local:
        if not settings.LOCAL_RESULTS_DIR.exists():
            return []
        # Scan LOCAL_RESULTS_DIR for run_* directories
        for path in settings.LOCAL_RESULTS_DIR.iterdir():
            if path.is_dir() and path.name.startswith("run_"):
                details = get_run_details(path.name)
                if details:
                    runs.append(details)
    else:
        try:
            client = _get_storage_client()
            bucket = client.bucket(settings.RESULTS_BUCKET)
            iterator = bucket.list_blobs(prefix="runs/", delimiter="/")
            list(iterator)
            for prefix in iterator.prefixes:
                # prefix is like "runs/run_uuid/"
                run_id = prefix.split("/")[-2]
                details = get_run_details(run_id)
                if details:
                    runs.append(details)
        except Exception as e:
            logger.error(f"Failed to list runs from GCS: {e}")
    return sorted(runs, key=lambda x: x.get("created_at", ""), reverse=True)

def get_run_details(run_id: str) -> Optional[dict]:
    """Retrieve and merge all JSON result files for a given run ID."""
    run_info = {
        "run_id": run_id,
        "status": "unknown",
        "created_at": None,
        "completed_at": None,
        "results": {}
    }

    if settings.is_local:
        run_dir = settings.LOCAL_RESULTS_DIR / run_id
        if not run_dir.exists():
            return None
            
        # Check active_run.json if it exists (active or crashed run)
        active_run_file = run_dir / "active_run.json"
        if active_run_file.exists():
            try:
                with open(active_run_file) as f:
                    active_meta = json.load(f)
                    status = active_meta.get("status", "running")
                    if status in ["running", "pending"]:
                        pid = active_meta.get("pid", 0)
                        if is_gbench_process_running(pid):
                            run_info["status"] = "running"
                        else:
                            active_meta["status"] = "failed"
                            run_info["status"] = "failed"
                            with open(active_run_file, "w") as wf:
                                json.dump(active_meta, wf, indent=2)
                    else:
                        run_info["status"] = status
                    
                    run_info["total_scenarios"] = active_meta.get("total_scenarios", 10)
                    run_info["created_at"] = active_meta.get("created_at")
                    run_info["tags"] = active_meta.get("tags", [])
                    run_info["model_name"] = active_meta.get("model_id")
            except Exception as e:
                logger.error(f"Failed to read active_run.json for {run_id}: {e}")

        # Read metadata.json if exists
        metadata_files = list(run_dir.glob("**/metadata.json"))
        if metadata_files:
            try:
                with open(metadata_files[0]) as f:
                    meta = json.load(f)
                    run_info.update(meta)
            except Exception as e:
                logger.error(f"Failed to read local metadata {metadata_files[0]}: {e}")

        # Read result files
        json_files = list(run_dir.glob("**/*.json"))
        
        for path in json_files:
            if path.name in ["openclaw.json", "qa-suite-summary.json", "metadata.json"]:
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                    _merge_result_data(run_info, data, path)
            except Exception as e:
                logger.error(f"Failed to read local result {path}: {e}")
        
        # Estimate created_at from directory creation time if missing
        if not run_info.get("created_at"):
            mtime = os.path.getmtime(run_dir)
            run_info["created_at"] = datetime.datetime.fromtimestamp(mtime).isoformat() + "Z"
            
    else:
        try:
            client = _get_storage_client()
            bucket = client.bucket(settings.RESULTS_BUCKET)
            blobs = list(bucket.list_blobs(prefix=f"runs/{run_id}/"))
            if not blobs:
                return None
                
            for blob in blobs:
                if blob.name.endswith("metadata.json"):
                    try:
                        content = blob.download_as_text()
                        meta = json.loads(content)
                        run_info.update(meta)
                    except Exception as e:
                        logger.error(f"Failed to read GCS metadata {blob.name}: {e}")
                elif blob.name.endswith(".json") and not blob.name.endswith("openclaw.json") and not blob.name.endswith("qa-suite-summary.json"):
                    try:
                        content = blob.download_as_text()
                        data = json.loads(content)
                        _merge_result_data(run_info, data, Path(blob.name))
                    except Exception as e:
                        logger.error(f"Failed to read GCS result {blob.name}: {e}")
                        
            # Estimate created_at from first blob updated time if missing
            if not run_info["created_at"] and blobs:
                run_info["created_at"] = blobs[0].updated.isoformat()
        except Exception as e:
            logger.error(f"Failed to fetch run {run_id} from GCS: {e}")
            return None

    # Derive overall status from merged results if not already failed/cancelled
    if run_info["status"] not in ["failed", "cancelled"] and run_info["results"]:
        run_info["status"] = "completed"
        # Find latest completed_at
        completed_times = [r.get("completed_at") for r in run_info["results"].values() if isinstance(r, dict) and r.get("completed_at")]
        if completed_times:
            run_info["completed_at"] = max(completed_times)
    
    return run_info

def _merge_result_data(run_info: dict, data: dict, path: Path):
    """Helper to merge result details into unified run_info object."""
    if not isinstance(data, dict):
        return
        
    benchmark_type = data.get("benchmark_type")
    
    # Fallback to identify benchmark type from filename
    if not benchmark_type:
        name = path.name.lower()
        if "quality" in name:
            benchmark_type = "quality"
        elif "stress" in name:
            benchmark_type = "stress"
        elif "serving" in name:
            benchmark_type = "serving"
        elif "throughput" in name:
            benchmark_type = "throughput"
            
    if benchmark_type:
        run_info["results"][benchmark_type] = data
        # Propagate model details if present
        if "model_name" in data and not run_info.get("model_name"):
            run_info["model_name"] = data["model_name"]
            run_info["model_short"] = data.get("model_short")
            
def upload_run_folder_to_gcs(run_id: str, local_run_dir: Path) -> None:
    """Upload all result files from a local run directory to GCS."""
    if not local_run_dir.exists():
        return
        
    try:
        client = _get_storage_client()
        bucket = client.bucket(settings.RESULTS_BUCKET)
        
        for path in local_run_dir.glob("**/*"):
            if path.is_file():
                # Resolve GCS destination path: runs/{run_id}/...
                relative_path = path.relative_to(local_run_dir)
                blob_name = f"runs/{run_id}/{relative_path}"
                logger.info(f"Uploading local file {path} to GCS blob {blob_name}...")
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(path))
    except Exception as e:
        logger.error(f"Failed to upload run folder {run_id} to GCS: {e}")

def delete_run(run_id: str) -> None:
    """Delete run folder from local storage or GCS."""
    if settings.is_local:
        run_dir = settings.LOCAL_RESULTS_DIR / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
    else:
        try:
            client = _get_storage_client()
            bucket = client.bucket(settings.RESULTS_BUCKET)
            blobs = bucket.list_blobs(prefix=f"runs/{run_id}/")
            for blob in blobs:
                blob.delete()
        except Exception as e:
            logger.error(f"Failed to delete run {run_id} from GCS: {e}")

def is_gbench_process_running(pid: int) -> bool:
    """Checks if a process with specified PID exists and matches gbench or python cmdline."""
    if pid <= 0:
        return False
    try:
        # Check if process is alive in OS
        os.kill(pid, 0)
        
        # Double check /proc to prevent recycled PID matches
        proc_path = Path(f"/proc/{pid}/cmdline")
        if proc_path.exists():
            try:
                with open(proc_path, "rb") as f:
                    cmdline = f.read().lower()
                    if b"gbench" in cmdline or b"python" in cmdline or b"openclaw" in cmdline:
                        return True
            except Exception:
                pass
            return False # proc exists but failed validation
        return True # fallback if /proc doesn't exist
    except OSError:
        return False
