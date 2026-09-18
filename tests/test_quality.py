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

"""Unit tests for QualityBenchmarkRunner."""

import json
import os
import subprocess
from unittest import mock
import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners.quality import QualityBenchmarkRunner


@pytest.fixture
def base_config(tmp_path):
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir()
    config.gemmaclaw_commit = "main"
    config.initialize()
    return config


@pytest.fixture
def mock_model():
    model = ModelConfig(
        short_name="gemma-4-E2B-it",
        hf_model_id="google/gemma-4-E2B-it",
        total_params_b=5.2,
        is_moe=False,
    )
    return model


def setup_multi_file_mock(mock_open_func, file_data_map):
    """Configure mock_open to handle multiple files.
    
    file_data_map: dict of {filename_substring: read_data}
    returns: dict of {filename_substring: mock_handle}
    """
    handles = {}
    for k, v in file_data_map.items():
        # Create a mock_open for this file to get its handle
        m = mock.mock_open(read_data=v)
        handles[k] = m.return_value

    default_handle = mock.MagicMock()

    def open_side_effect(filename, mode='r', *args, **kwargs):
        for k, handle in handles.items():
            if k in str(filename):
                return handle
        return default_handle

    mock_open_func.side_effect = open_side_effect
    # Store default handle as well
    handles["default"] = default_handle
    return handles


@mock.patch("gbench.runners.quality.os.path.exists", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.Popen", autospec=True)
@mock.patch("gbench.runners.quality.shutil.rmtree", autospec=True)
@mock.patch("gbench.runners.quality.os.mkdir", autospec=True)
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_run_local_server(
    mock_open_file,
    mock_mkdir,
    mock_rmtree,
    mock_popen,
    mock_exists,
    base_config,
    mock_model,
):
    """Test standard execution starting local vLLM."""
    runner = QualityBenchmarkRunner(base_config)
    
    # Mock preflight and repo setup
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()
    
    # Mock exists to only return True for the summary file
    mock_exists.side_effect = lambda path: "qa-suite-summary.json" in str(path)

    # Setup multi-file mock
    handles = setup_multi_file_mock(mock_open_file, {
        "openclaw.json": "",
        "qa-suite-summary.json": '{"counts": {"total": 10, "passed": 8, "failed": 2, "blocked": 0}}',
        "quality_": "" # for results output
    })

    mock_process = mock.MagicMock(autospec=True)
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    result = runner.run(mock_model, ModelFormat.HF)

    runner._start_server.assert_called_once_with(mock_model, ModelFormat.HF)
    runner._cleanup_server.assert_called_once()

    # Verify openclaw.json content
    openclaw_handle = handles["openclaw.json"]
    written_data = "".join(call.args[0] for call in openclaw_handle.write.call_args_list)
    config_json = json.loads(written_data)
    
    assert config_json["models"]["providers"]["vllm"]["baseUrl"] == f"http://127.0.0.1:{runner.server_port}/v1"
    assert config_json["models"]["providers"]["vllm"]["apiKey"] == "local-vllm"
    assert config_json["plugins"]["entries"]["qa-lab"]["enabled"] is True

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    env = kwargs["env"]
    assert "openclaw" in cmd
    assert "qa" in cmd
    assert "suite" in cmd
    assert env["OPENCLAW_ENABLE_PRIVATE_QA_CLI"] == "1"
    assert env["VLLM_API_KEY"] == "local-vllm"

    assert result["passed_scenarios"] == 8
    assert result["total_scenarios"] == 10
    assert result["pass_rate"] == 80.0


@mock.patch("urllib.request.urlopen", side_effect=OSError("Metadata unavailable"))
@mock.patch("gbench.runners.quality.os.path.exists", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.Popen", autospec=True)
@mock.patch("gbench.runners.quality.shutil.rmtree", autospec=True)
@mock.patch("gbench.runners.quality.os.mkdir", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.check_output", autospec=True)
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch.dict("os.environ", {"VLLM_API_KEY": ""}, clear=False)
def test_run_remote_endpoint_google(
    mock_open_file,
    mock_check_output,
    mock_mkdir,
    mock_rmtree,
    mock_popen,
    mock_exists,
    mock_urlopen,
    base_config,
    mock_model,
):
    """Test remote endpoint targeting Cloud Run (fetches OIDC token)."""
    base_config.remote_endpoint = "https://gbench-vllm-abc123-uc.a.run.app/v1"
    runner = QualityBenchmarkRunner(base_config)
    
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock()
    runner._cleanup_server = mock.MagicMock()
    
    mock_exists.side_effect = lambda path: "qa-suite-summary.json" in str(path)

    # Setup multi-file mock
    handles = setup_multi_file_mock(mock_open_file, {
        "openclaw.json": "",
        "qa-suite-summary.json": '{"counts": {"total": 10, "passed": 8, "failed": 2, "blocked": 0}}',
        "quality_": ""
    })

    mock_process = mock.MagicMock(autospec=True)
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    mock_check_output.return_value = "mocked-oidc-token\n"

    # GCP identity-token auto-auth is opt-in (GBENCH_GCP_AUTH=1); this test exercises that path.
    with mock.patch.dict("os.environ", {"GBENCH_GCP_AUTH": "1"}):
        result = runner.run(mock_model, ModelFormat.HF)

    runner._start_server.assert_not_called()
    runner._cleanup_server.assert_called_once() # Called in finally block

    mock_check_output.assert_called_once_with(
        ["gcloud", "auth", "print-identity-token"], text=True
    )

    openclaw_handle = handles["openclaw.json"]
    written_data = "".join(call.args[0] for call in openclaw_handle.write.call_args_list)
    config_json = json.loads(written_data)
    assert config_json["models"]["providers"]["vllm"]["baseUrl"] == base_config.remote_endpoint
    assert config_json["models"]["providers"]["vllm"]["apiKey"] == "mocked-oidc-token"

    args, kwargs = mock_popen.call_args
    env = kwargs["env"]
    assert env["VLLM_API_KEY"] == "mocked-oidc-token"

    assert result["passed_scenarios"] == 8
    assert result["total_scenarios"] == 10
    assert result["pass_rate"] == 80.0


@mock.patch("gbench.runners.quality.os.path.exists", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.Popen", autospec=True)
@mock.patch("gbench.runners.quality.shutil.rmtree", autospec=True)
@mock.patch("gbench.runners.quality.os.mkdir", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.check_output", autospec=True)
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_run_remote_endpoint_google_with_key_override(
    mock_open_file,
    mock_check_output,
    mock_mkdir,
    mock_rmtree,
    mock_popen,
    mock_exists,
    base_config,
    mock_model,
    monkeypatch,
):
    """Test remote endpoint but VLLM_API_KEY is already set (bypasses gcloud)."""
    base_config.remote_endpoint = "https://gbench-vllm-abc123-uc.a.run.app/v1"
    runner = QualityBenchmarkRunner(base_config)
    
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock()
    runner._cleanup_server = mock.MagicMock()
    
    mock_exists.side_effect = lambda path: "qa-suite-summary.json" in str(path)

    # Setup multi-file mock
    handles = setup_multi_file_mock(mock_open_file, {
        "openclaw.json": "",
        "qa-suite-summary.json": '{"counts": {"total": 10, "passed": 8, "failed": 2, "blocked": 0}}',
        "quality_": ""
    })

    mock_process = mock.MagicMock(autospec=True)
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    monkeypatch.setenv("VLLM_API_KEY", "user-override-key")

    result = runner.run(mock_model, ModelFormat.HF)

    mock_check_output.assert_not_called()
    runner._cleanup_server.assert_called_once()

    openclaw_handle = handles["openclaw.json"]
    written_data = "".join(call.args[0] for call in openclaw_handle.write.call_args_list)
    config_json = json.loads(written_data)
    assert config_json["models"]["providers"]["vllm"]["apiKey"] == "user-override-key"

    assert result["passed_scenarios"] == 8
    assert result["total_scenarios"] == 10
    assert result["pass_rate"] == 80.0


@mock.patch("gbench.runners.quality.os.path.exists", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.Popen", autospec=True)
@mock.patch("gbench.runners.quality.shutil.rmtree", autospec=True)
@mock.patch("gbench.runners.quality.os.mkdir", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.check_output", autospec=True)
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_run_remote_endpoint_google_no_auto_auth_without_optin(
    mock_open_file,
    mock_check_output,
    mock_mkdir,
    mock_rmtree,
    mock_popen,
    mock_exists,
    base_config,
    mock_model,
    monkeypatch,
):
    """A GCP-looking endpoint with no key and GBENCH_GCP_AUTH unset must NOT shell out to gcloud
    or the metadata server (auto-auth is opt-in); it falls back to the local sentinel key."""
    base_config.remote_endpoint = "https://gbench-vllm-abc123-uc.a.run.app/v1"
    runner = QualityBenchmarkRunner(base_config)

    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock()
    runner._cleanup_server = mock.MagicMock()

    mock_exists.side_effect = lambda path: "qa-suite-summary.json" in str(path)
    handles = setup_multi_file_mock(mock_open_file, {
        "openclaw.json": "",
        "qa-suite-summary.json": '{"counts": {"total": 10, "passed": 8, "failed": 2, "blocked": 0}}',
        "quality_": ""
    })
    mock_process = mock.MagicMock(autospec=True)
    mock_process.wait.return_value = 0
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("GBENCH_GCP_AUTH", raising=False)

    result = runner.run(mock_model, ModelFormat.HF)

    mock_check_output.assert_not_called()   # no gcloud subprocess when opt-in is off
    openclaw_handle = handles["openclaw.json"]
    written_data = "".join(call.args[0] for call in openclaw_handle.write.call_args_list)
    config_json = json.loads(written_data)
    assert config_json["models"]["providers"]["vllm"]["apiKey"] == "local-vllm"
    assert result["pass_rate"] == 80.0


@mock.patch("gbench.runners.quality.os.path.exists", autospec=True)
@mock.patch("gbench.runners.quality.subprocess.Popen", autospec=True)
@mock.patch("gbench.runners.quality.shutil.rmtree", autospec=True)
@mock.patch("gbench.runners.quality.os.mkdir", autospec=True)
@mock.patch("builtins.open", new_callable=mock.mock_open)
def test_run_timeout_kills_process(
    mock_open_file,
    mock_mkdir,
    mock_rmtree,
    mock_popen,
    mock_exists,
    base_config,
    mock_model,
):
    """Test that timeout kills the running process and raises RuntimeError."""
    runner = QualityBenchmarkRunner(base_config)
    
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()

    mock_exists.side_effect = lambda path: "qa-suite-summary.json" in str(path)

    # Setup multi-file mock (even though it will fail before reading summary,
    # it still writes openclaw.json)
    handles = setup_multi_file_mock(mock_open_file, {
        "openclaw.json": "",
        "qa-suite-summary.json": '{"counts": {"total": 10, "passed": 8, "failed": 2, "blocked": 0}}',
        "quality_": ""
    })

    mock_process = mock.MagicMock(autospec=True)
    mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd=["openclaw"], timeout=3600), -9]
    mock_popen.return_value = mock_process

    with pytest.raises(RuntimeError, match="gemmaclaw suite timed out"):
        runner.run(mock_model, ModelFormat.HF)

    mock_process.kill.assert_called_once()
    assert mock_process.wait.call_count == 2
    runner._cleanup_server.assert_called_once()


def test_node_version_parses_and_handles_missing():
    """_node_version returns (0,0) for a missing/None binary and a parsed
    (major, minor) for a real one, so the nvm-upgrade gate can compare it."""
    import shutil
    from gbench.runners.quality import _node_version
    assert _node_version(None) == (0, 0)
    assert _node_version("/nonexistent/definitely-not-node") == (0, 0)
    real = shutil.which("node")
    if real:
        mj, mn = _node_version(real)
        assert isinstance(mj, int) and isinstance(mn, int)


def test_git_rewrite_safe_env_blanks_global_config(base_config, monkeypatch):
    """The quality git env neutralizes the host github HTTPS->SSH rewrite by
    default (mirrors the swe_lancer fix) and forbids interactive git prompts."""
    monkeypatch.delenv("GBENCH_QUALITY_GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    runner = QualityBenchmarkRunner(base_config)
    env = runner._git_rewrite_safe_env()
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]


def test_git_rewrite_safe_env_override(base_config, monkeypatch):
    """A host that legitimately needs its rewrites can point the knob at a real
    gitconfig instead of /dev/null."""
    monkeypatch.setenv("GBENCH_QUALITY_GIT_CONFIG_GLOBAL", "/custom/gitconfig")
    runner = QualityBenchmarkRunner(base_config)
    env = runner._git_rewrite_safe_env()
    assert env["GIT_CONFIG_GLOBAL"] == "/custom/gitconfig"
