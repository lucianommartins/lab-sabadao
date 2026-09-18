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

"""Centralized log and results management.

This module provides a LogManager class that handles all path generation,
directory creation, and file I/O for benchmark results and logs.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    """json.dump fallback for values the stdlib encoder cannot handle.

    Result dicts routinely carry numpy scalars - e.g. a stress level's
    ``passed`` is ``numpy.bool_`` (a numpy-float TTFT compared to the
    threshold) and stats are ``numpy.float64``. numpy 2.x names its bool
    scalar ``bool``, so json raises "Object of type bool is not JSON
    serializable". numpy scalars expose ``.item()``, which returns the
    native Python equivalent (bool/int/float) so numbers stay numbers;
    anything else falls back to ``str`` rather than crashing the write.
    """
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(o)


class LogManager:
    """Centralized manager for benchmark results and logs.
    
    Handles:
    - Timestamped results directory creation
    - Path generation for results and logs
    - File I/O for saving/loading results
    - Directory structure management
    """
    
    def __init__(self, base_dir: Path | str = "./results"):
        """Initialize LogManager with timestamped results directory.
        
        Args:
            base_dir: Base directory for results (default: ./results)
        """
        self.base_dir = Path(base_dir)
        
        # If base_dir is already an existing run directory (contains metadata.json or eval_*.json), reuse it directly
        if self.base_dir.exists() and (
            (self.base_dir / "metadata.json").exists()
            or any(self.base_dir.glob("eval_*.json"))
        ):
            self.results_dir = self.base_dir
            self.timestamp = self.base_dir.name
        else:
            self.timestamp = datetime.now().strftime("%m-%d-%H%M%S")
            self.results_dir = self.base_dir / self.timestamp
        
        # Create directory structure
        self._setup_directories()

        # Expose the resolved run directory so eval runners that write their own auxiliary
        # artifacts (e.g. tau2's per-task simulation traces) can co-locate them with the
        # eval JSONs without threading the path through every eval signature.
        os.environ["GBENCH_RESULTS_DIR"] = str(self.results_dir)

        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def _setup_directories(self):
        """Create the directory structure for results."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "performance").mkdir(exist_ok=True)
        (self.results_dir / "quality").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
    
    # === Path Generation ===
    
    def get_serving_result_path(
        self,
        model_name: str,
        format: str,
        batch_size: int,
        multimodal: bool = False,
    ) -> Path:
        """Generate path for serving benchmark result file."""
        suffix = "_mm" if multimodal else ""
        filename = f"serve_{model_name}_{format}_b{batch_size}{suffix}.json"
        return self.results_dir / "performance" / filename
    
    def get_serving_log_path(
        self,
        model_name: str,
        format: str,
        batch_size: int,
        multimodal: bool = False,
    ) -> Path:
        """Generate path for serving benchmark log file."""
        suffix = "_mm" if multimodal else ""
        filename = f"serve_{model_name}_{format}_b{batch_size}{suffix}.log"
        return self.results_dir / "logs" / filename

    def get_server_log_path(self, model_name: str, format: str) -> Path:
        """Generate path for vLLM server stdout/stderr logs.
        
        Unlike benchmark logs, this is for the server process itself,
        which is shared across multiple benchmark configurations.
        """
        filename = f"server_{model_name}_{format}.log"
        return self.results_dir / "logs" / filename
    
    def get_throughput_result_path(
        self,
        model_name: str,
        format: str,
        input_length: int,
        output_length: int,
        batch_size: int,
    ) -> Path:
        """Generate path for throughput benchmark result file."""
        filename = (
            f"throughput_{model_name}_{format}_"
            f"in{input_length}_out{output_length}_b{batch_size}.json"
        )
        return self.results_dir / "performance" / filename
    
    def get_throughput_log_path(
        self,
        model_name: str,
        format: str,
        input_length: int,
        output_length: int,
        batch_size: int,
    ) -> Path:
        """Generate path for throughput benchmark log file."""
        filename = (
            f"throughput_{model_name}_{format}_"
            f"in{input_length}_out{output_length}_b{batch_size}.log"
        )
        return self.results_dir / "logs" / filename
    
    def get_stress_result_path(
        self,
        model_name: str,
        format: str,
        mode: str,
    ) -> Path:
        """Generate path for stress test result file.
        
        Args:
            model_name: Model short name
            format: Model format (hf, gguf)
            mode: Test mode ('text' or 'mm')
        """
        filename = f"stress_{model_name}_{format}_{mode}.json"
        return self.results_dir / "performance" / filename
    
    def get_quality_result_path(
        self,
        model_name: str,
        format: str,
    ) -> Path:
        """Generate path for quality benchmark result file."""
        filename = f"quality_{model_name}_{format}.json"
        return self.results_dir / "quality" / filename

    def get_quality_log_path(
        self,
        model_name: str,
        format: str,
    ) -> Path:
        """Generate path for quality benchmark log file."""
        filename = f"quality_{model_name}_{format}.log"
        return self.results_dir / "logs" / filename
    
    # === File I/O ===
    
    def save_result(self, path: Path, data: dict) -> None:
        """Save benchmark result to JSON file.
        
        Args:
            path: Path to save file
            data: Result data dictionary
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        logger.debug(f"Saved result to: {path}")

    def save_metadata(self, metadata: dict) -> None:
        """Save run metadata to metadata.json."""
        path = self.results_dir / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)
        logger.debug(f"Saved run metadata to: {path}")

    def save_summary(self, summary_data: dict) -> Path:
        """Save aggregated run summary to summary.json."""
        path = self.results_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=_json_default)
        logger.debug(f"Saved aggregated summary to: {path}")
        return path

    def save_csv(self, filename: str, rows: list[dict], fieldnames: list[str]) -> Path:
        """Save rows to CSV file in results directory."""
        import csv
        path = self.results_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.debug(f"Saved CSV report to: {path}")
        return path
    
    def load_result(self, path: Path) -> Optional[dict]:
        """Load benchmark result from JSON file.
        
        Args:
            path: Path to result file
            
        Returns:
            Result data dictionary, or None if file doesn't exist
        """
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    
    def save_log(self, path: Path, stdout: str, stderr: str) -> None:
        """Save benchmark log to file.
        
        Args:
            path: Path to log file
            stdout: Standard output content
            stderr: Standard error content
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"=== STDOUT ===\n{stdout}\n")
            f.write(f"=== STDERR ===\n{stderr}\n")
        logger.debug(f"Saved log to: {path}")
    
    def result_exists(self, path: Path) -> bool:
        """Check if a result file already exists."""
        return path.exists()
    
    # === Summary Methods ===
    
    def list_result_files(self) -> list[Path]:
        """List all result files in the performance directory."""
        perf_dir = self.results_dir / "performance"
        if not perf_dir.exists():
            return []
        return sorted(perf_dir.glob("*.json"))
    
    def list_quality_result_files(self) -> list[Path]:
        """List all result files in the quality directory."""
        qual_dir = self.results_dir / "quality"
        if not qual_dir.exists():
            return []
        return sorted(qual_dir.glob("*.json"))
    
    def list_log_files(self) -> list[Path]:
        """List all log files in the logs directory."""
        logs_dir = self.results_dir / "logs"
        if not logs_dir.exists():
            return []
        return sorted(logs_dir.glob("*.log"))
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all results in this run.
        
        Returns:
            Dictionary with result counts and paths
        """
        perf_files = self.list_result_files()
        qual_files = self.list_quality_result_files()
        result_files = perf_files + qual_files
        log_files = self.list_log_files()
        
        return {
            "results_dir": str(self.results_dir.absolute()),
            "timestamp": self.timestamp,
            "result_count": len(result_files),
            "log_count": len(log_files),
            "result_files": [f.name for f in result_files],
            "log_files": [f.name for f in log_files],
        }
