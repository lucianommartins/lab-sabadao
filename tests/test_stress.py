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

"""Stress runner regression tests."""

import json
import urllib.request

from gbench.core.config import BenchmarkConfig
from gbench.runners.stress import StressTestRunner, _delta_piece
from gbench.runners.serving import ServingBenchmarkRunner


def test_delta_piece_counts_reasoning_tokens():
    """A reasoning model (e.g. gemma via Ollama) streams its reply as
    reasoning/reasoning_content deltas. The stress SSE parser must count those as
    real decode - counting only `content` made every request look empty and the
    whole sweep skip with 'no usable stream'."""
    # chat deltas
    assert _delta_piece({"delta": {"content": "hi "}}, "delta") == "hi "
    assert _delta_piece({"delta": {"reasoning": "think "}}, "delta") == "think "
    assert _delta_piece({"delta": {"reasoning_content": "rc "}}, "delta") == "rc "
    assert _delta_piece({"delta": {}}, "delta") == ""            # empty -> not counted
    assert _delta_piece({"delta": {"role": "assistant"}}, "delta") == ""
    # legacy /v1/completions text field
    assert _delta_piece({"text": "tok "}, "text") == "tok "
    assert _delta_piece({}, "text") == ""


def test_client_pool_multiproc_is_not_dead_code():
    """`self._mp_ctx` used to be initialized AFTER a `return` in _images_per_request()
    (unreachable), so `_get_client_pool()` raised AttributeError and every rate point
    silently fell back to a single in-process client while the artifact still reported
    `num_client_procs=8` / method=...multiproc / "contention-free". This asserts the
    context is set at construction and the pool spawns real worker processes."""
    r = StressTestRunner(BenchmarkConfig())
    assert r._mp_ctx is not None                       # was never set (the bug)
    assert r._mp_ctx.get_start_method() == "spawn"
    r._num_client_procs = 2                             # keep the spawn test light
    pool = r._get_client_pool()
    try:
        # Workers actually spawn and execute (would raise on the dead-code bug).
        assert list(pool.map(abs, [-5, -6])) == [5, 6]
    finally:
        r._shutdown_client_pool()
    assert r._client_pool is None


def _text_url(remote_endpoint, local_port=8000, monkeypatch=None):
    """Build the stress TEXT api_url for a given endpoint via the real code path."""
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = remote_endpoint
    r = StressTestRunner(cfg)
    r._api_model_id = "m"
    r._multimodal = False
    if remote_endpoint is None:
        r._serving_runner = type("S", (), {"server_port": local_port})()
    # Stub the heavy request generation; we only care about the URL.
    r._generate_requests = lambda n, model=None, format=None: [type("R", (), {"prompt": "hi"})()]
    api_url, _field, _payloads = r._build_payloads(model=None, format=None, num_prompts=1)
    return api_url


def _mm_url(remote_endpoint):
    """Build the stress MULTIMODAL api_url via the real code path (num_prompts=0 skips image IO)."""
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = remote_endpoint
    r = StressTestRunner(cfg)
    r._api_model_id = "m"
    r._multimodal = True
    api_url, _field, _payloads = r._build_payloads(model=None, format=None, num_prompts=0)
    return api_url


def test_remote_endpoint_with_v1_is_not_doubled():
    """--remote-endpoint <URL>/v1 (the documented convention) must not become /v1/v1/...

    Regression: the stress runner blindly appended `/v1/completions` to a base_url that
    already ended in `/v1`, so every request 404'd (completed=0/failed=N at every rate,
    goodput 0.00 req/s) while serving - which normalizes the same endpoint - worked fine.
    """
    assert _text_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/completions"
    assert _mm_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/chat/completions"
    # Trailing slash on the /v1 base must also normalize.
    assert _text_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1/completions"


def test_remote_endpoint_without_v1_gets_v1_appended():
    assert _text_url("http://host:9000") == "http://host:9000/v1/completions"
    assert _mm_url("http://host:9000") == "http://host:9000/v1/chat/completions"


def test_local_serving_url_gets_v1_appended():
    assert _text_url(None, local_port=8123) == "http://127.0.0.1:8123/v1/completions"


def test_stress_max_qps_and_prompts_default_env_flag_precedence(monkeypatch):
    """The default safety cap (8192 QPS) + prompt cap (8000) are high enough that a fast
    multi-GPU server finds its real knee out of the box, and stay overridable. Precedence:
    config flag > GBENCH_STRESS_* env > class default."""
    monkeypatch.delenv("GBENCH_STRESS_MAX_QPS", raising=False)
    monkeypatch.delenv("GBENCH_STRESS_MAX_PROMPTS", raising=False)
    r = StressTestRunner(BenchmarkConfig())
    assert r.max_qps == StressTestRunner.MAX_QPS == 8192.0
    assert r.max_prompts == StressTestRunner.MAX_PROMPTS == 8000
    # Client-proc ceiling raised in lockstep so the load generator can actually OFFER 8192 QPS on a
    # big host (the runtime value still auto-caps to cpu_count-1). Assert the class constant, which
    # is machine-independent (the computed _num_client_procs depends on the CI host's core count).
    assert StressTestRunner.NUM_CLIENT_PROCS == 32

    monkeypatch.setenv("GBENCH_STRESS_MAX_QPS", "4096")
    monkeypatch.setenv("GBENCH_STRESS_MAX_PROMPTS", "8000")
    r_env = StressTestRunner(BenchmarkConfig())
    assert r_env.max_qps == 4096.0
    assert r_env.max_prompts == 8000

    cfg = BenchmarkConfig()
    cfg.stress_max_qps = 2048
    cfg.stress_max_prompts = 5000
    r_flag = StressTestRunner(cfg)          # flag set -> wins over the env above
    assert r_flag.max_qps == 2048.0
    assert r_flag.max_prompts == 5000


def test_stress_client_procs_and_reps_flag_env_precedence(monkeypatch):
    """--stress-client-procs / --stress-reps are uniform with the new knobs:
    flag > GBENCH_STRESS_* env > default (client procs stay auto-capped to cpu_count-1)."""
    monkeypatch.delenv("GBENCH_STRESS_CLIENT_PROCS", raising=False)
    monkeypatch.delenv("GBENCH_STRESS_REPS", raising=False)
    r = StressTestRunner(BenchmarkConfig())
    assert r._reps == StressTestRunner.STRESS_REPS == 3

    # small values so the cpu_count-1 cap never masks the assertion
    monkeypatch.setenv("GBENCH_STRESS_CLIENT_PROCS", "2")
    monkeypatch.setenv("GBENCH_STRESS_REPS", "5")
    r_env = StressTestRunner(BenchmarkConfig())
    assert r_env._num_client_procs == 2
    assert r_env._reps == 5

    cfg = BenchmarkConfig()
    cfg.stress_client_procs = 3
    cfg.stress_reps = 7
    r_flag = StressTestRunner(cfg)          # flag wins over the env above
    assert r_flag._num_client_procs == 3
    assert r_flag._reps == 7


def test_mm_effective_input_len_tracks_probed_vision_tokens():
    """The reported MM effective input length must track the per-image soft-token
    count actually charged by the server, not a hardcoded 280. gemma-4 served with
    max_soft_tokens=1120 (--hf-overrides) charges ~4x the class default; the old
    constant under-reported the MM prefill ~3.7x (128+4*280=1248 vs 128+4*1120=4608)
    and mis-sized the padded-text budget."""
    cfg = BenchmarkConfig()
    cfg.max_model_len = 262144   # large context so the text budget is not clamped
    r = StressTestRunner(cfg)
    assert r._images_per_request() == 4
    # Class default (280): 128 text + 4*280 images.
    r._vision_tokens_per_image = r.PER_IMAGE_SOFT_TOKENS
    assert r._mm_effective_input_len(128, 128) == 128 + 4 * 280 == 1248
    # Server-measured (gemma-4 deployed, 1120): 128 text + 4*1120 images.
    r._vision_tokens_per_image = 1120
    assert r._mm_effective_input_len(128, 128) == 128 + 4 * 1120 == 4608


def test_mm_text_budget_clamps_when_images_fill_context():
    """At the uniform 4096 context, 4 images at 1120 tok/img (4480) already exceed
    the window, so the padded-text budget must clamp to the MIN floor rather than go
    negative. This only surfaces once the true (probed) per-image count is used; with
    the old 280 the images fit and the clamp never engaged (masking the overflow)."""
    cfg = BenchmarkConfig()          # default max_model_len -> uniform 4096
    r = StressTestRunner(cfg)
    r._vision_tokens_per_image = 1120
    # 4*1120=4480 > 4096 -> text clamps to MIN_MM_TEXT_TOKENS, not a negative value.
    assert r._mm_text_tokens(128, 128) == r.MIN_MM_TEXT_TOKENS
    assert r._mm_effective_input_len(128, 128) == r.MIN_MM_TEXT_TOKENS + 4 * 1120
    # With the old 280 default the images fit, so text stays at the full 128.
    r._vision_tokens_per_image = 280
    assert r._mm_text_tokens(128, 128) == 128


def test_mm_preflight_returns_server_measured_per_image(monkeypatch):
    """_assert_mm_images_processed returns the SERVER-MEASURED per-image soft-token
    gain = (img prompt_tokens - text prompt_tokens) / n_img, so the stress runner can
    size + report the MM workload against what the server actually charges (gemma-4:
    1120 via --hf-overrides) instead of the 280 default. Robust to any backend."""
    cfg = BenchmarkConfig()          # remote_endpoint None -> file:// image URLs, no file IO
    r = ServingBenchmarkRunner(cfg)
    r._mm_image_paths = ["/tmp/probe_never_opened.jpg"]   # non-empty so the probe runs
    r._images_per_request = lambda: 4

    class _Resp:
        def __init__(self, ptok):
            self._b = json.dumps({"usage": {"prompt_tokens": ptok}}).encode("utf-8")
        def read(self):
            return self._b
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    # text-only = 10 tok; +4 images = 10 + 4*1120 -> per_img = 1120
    vals = iter([10, 10 + 4 * 1120])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(next(vals)))
    got = r._assert_mm_images_processed("m", "http://127.0.0.1:8000", per_img_tokens=280)
    assert got == 1120

    # Skipped-probe path (no images) returns None, so callers keep the default.
    r._mm_image_paths = []
    assert r._assert_mm_images_processed("m", "http://127.0.0.1:8000") is None


def test_hist_delta_p99_ms_from_histogram_delta():
    """P99 of the inter-token-latency of requests that ran between two cumulative
    histogram snapshots; None when a snapshot is missing or the window is empty."""
    before = ({0.01: 0.0, 0.02: 0.0, float("inf"): 0.0}, 0.0)
    after = ({0.01: 90.0, 0.02: 99.0, float("inf"): 100.0}, 100.0)
    assert StressTestRunner._hist_delta_p99_ms(before, after) == 20.0   # le=0.02 -> 20ms
    assert StressTestRunner._hist_delta_p99_ms(None, after) is None
    assert StressTestRunner._hist_delta_p99_ms(before, before) is None  # 0 tokens in window


def test_metrics_url_strips_v1():
    """/metrics lives at the server ROOT; a /v1 endpoint must not become /v1/metrics."""
    cfg = BenchmarkConfig(); cfg.remote_endpoint = "http://127.0.0.1:8000/v1"
    assert StressTestRunner(cfg)._metrics_url() == "http://127.0.0.1:8000/metrics"
    cfg2 = BenchmarkConfig(); cfg2.remote_endpoint = "http://host:9000"
    assert StressTestRunner(cfg2)._metrics_url() == "http://host:9000/metrics"
    r = StressTestRunner(BenchmarkConfig())              # local serve, no endpoint
    r._serving_runner = type("S", (), {"server_port": 8123})()
    assert r._metrics_url() == "http://127.0.0.1:8123/metrics"


def test_server_itl_snapshot_parses_inter_token_latency(monkeypatch):
    """Parses vllm:inter_token_latency_seconds buckets/count summed across DP engines."""
    metrics = "\n".join([
        'vllm:inter_token_latency_seconds_bucket{engine="0",le="0.01"} 40.0',
        'vllm:inter_token_latency_seconds_bucket{engine="1",le="0.01"} 50.0',
        'vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf"} 45.0',
        'vllm:inter_token_latency_seconds_bucket{engine="1",le="+Inf"} 55.0',
        'vllm:inter_token_latency_seconds_count{engine="0"} 45.0',
        'vllm:inter_token_latency_seconds_count{engine="1"} 55.0',
    ])
    class _R:
        def read(self): return metrics.encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    cfg = BenchmarkConfig(); cfg.remote_endpoint = "http://127.0.0.1:8000/v1"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _R())
    snap = StressTestRunner(cfg)._server_itl_snapshot()
    assert snap is not None
    buckets, count = snap
    assert count == 100.0
    assert buckets[0.01] == 90.0 and buckets[float("inf")] == 100.0


def _rate_metrics(**over):
    base = dict(achieved_qps=3.0, arrival_qps=3.0, slo_attainment=1.0,
                p99_ttft_ms=60.0, p99_tpot_ms=8.0, p99_itl_ms=800.0,
                p99_itl_server_ms=25.0, mean_e2e_ms=1000.0, steady_n=500,
                completed=200, failed=0, client_bound=False,
                actual_send_qps=3.0, client_sched_lag_s=0.0, num_client_procs=8)
    base.update(over); return base


def test_knee_gates_on_server_itl_not_client_confound(monkeypatch):
    """The MM confound: client P99 ITL 800ms (would trip the 200ms SLO) but the
    server's own P99 ITL is 25ms -> the rate is SUSTAINABLE. Gating on server ITL
    fixes the ~7x MM knee under-report; without /metrics it correctly falls back to
    the (failing) client ITL; a genuinely high server ITL still fails."""
    r = StressTestRunner(BenchmarkConfig())
    # (a) server ITL healthy -> PASS despite confounded client ITL
    monkeypatch.setattr(r, "_measure_rate_point", lambda *a, **k: _rate_metrics())
    passed, *_ = r._test_rate(3.0, None, None)
    assert passed is True
    assert r._tested_rates[3.0]["itl_slo_source"] == "server"
    # (b) no server metrics -> fall back to client ITL (800ms) -> FAIL
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _rate_metrics(p99_itl_server_ms=None))
    passed_b, *_ = r._test_rate(3.0, None, None)
    assert passed_b is False
    assert r._tested_rates[3.0]["itl_slo_source"] == "client"
    # (c) server ITL genuinely high -> real preemption -> FAIL
    r._tested_rates = {}
    monkeypatch.setattr(r, "_measure_rate_point",
                        lambda *a, **k: _rate_metrics(p99_itl_server_ms=350.0))
    passed_c, *_ = r._test_rate(3.0, None, None)
    assert passed_c is False
    assert r._tested_rates[3.0]["itl_slo_source"] == "server"


class _StubModel:
    short_name = "stub"
    name = "stub-model"


class _StubFormat:
    value = "hf"


def _preflight_runner(monkeypatch, ttft_ms):
    """A StressTestRunner wired so _preflight_floor_check needs no server/tokenizer:
    _build_payloads returns dummy payloads, and every single-stream probe returns a
    record with the given TTFT."""
    r = StressTestRunner(BenchmarkConfig(), ttft_threshold_ms=5000, tpot_threshold_ms=200)
    monkeypatch.setattr(r, "_build_payloads",
                        lambda *a, **k: ("http://x/v1/completions", "text",
                                         [{"i": 1}, {"i": 2}, {"i": 3}]))
    monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 128, 128))
    monkeypatch.setattr(
        "gbench.runners.stress._stress_client_worker",
        lambda task: {"records": [{"ok": True, "ttft": ttft_ms, "send": 0.0,
                                   "done": ttft_ms / 1000.0 + 1.0}], "sched_lag_s": 0.0})
    return r


def test_preflight_bails_below_floor(monkeypatch):
    """Single-stream TTFT above the SLO -> no open-loop rate can pass -> early-exit
    result (below_stress_floor), NOT a harness failure."""
    r = _preflight_runner(monkeypatch, ttft_ms=20000.0)   # 20s >> 5000ms SLO
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out is not None
    assert out["below_stress_floor"] is True
    assert out["max_sustainable_qps"] == 0.0
    assert out["slo_met"] is False
    assert out["preflight_single_stream_ttft_ms"] == 20000.0
    assert not out.get("failed")            # NOT routed to the failure bucket / exit code
    assert out["multimodal"] is False


def test_preflight_proceeds_when_single_stream_meets_slo(monkeypatch):
    """Single-stream TTFT under the SLO -> return None so the normal sweep runs."""
    r = _preflight_runner(monkeypatch, ttft_ms=120.0)     # 120ms << 5000ms SLO
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out is None


def test_preflight_bails_when_endpoint_returns_no_stream(monkeypatch):
    """If not one single-stream request produces output tokens, the request path is
    broken for this endpoint (e.g. 0 completions) -> bail fast with a diagnostic
    instead of falling through to a doomed sweep."""
    r = StressTestRunner(BenchmarkConfig(), ttft_threshold_ms=5000, tpot_threshold_ms=200)
    monkeypatch.setattr(r, "_build_payloads",
                        lambda *a, **k: ("http://x/v1/completions", "text", [{"i": 1}]))
    monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 128, 128))
    monkeypatch.setattr("gbench.runners.stress._stress_client_worker",
                        lambda task: {"records": [{"ok": False, "ttft": None}], "sched_lag_s": 0.0})
    out = r._preflight_floor_check(_StubModel(), _StubFormat(), multimodal=False)
    assert out is not None
    assert out["preflight_no_stream"] is True
    assert out["max_sustainable_qps"] == 0.0
    assert not out.get("failed")


def test_build_payloads_remote_uses_chat_with_coherent_prompt(monkeypatch):
    """Remote endpoints must use /v1/chat/completions + a coherent prompt (mirroring
    the serving pillar) so a backend that ignores ignore_eos (e.g. Ollama) still emits
    tokens - not the random /v1/completions prompt that draws an immediate EOS."""
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = "http://localhost:11434/v1"
    r = StressTestRunner(cfg)
    r._tokenizer = object()
    r._api_model_id = "gemma4-qat:4b"

    class _SR:
        NOTHINK_SYSTEM = "Answer directly."
        def _build_text_prompt(self, in_len, out_len, tokenizer, seed=0):
            return f"coherent prompt {seed}"
        def _resolve_model_id(self, *a, **k):
            return "gemma4-qat:4b"
    r._serving_runner = _SR()
    monkeypatch.setattr(r, "_workload_shape", lambda: ("random", 128, 128))

    url, field, payloads = r._build_payloads(model=None, format=None, num_prompts=2)
    assert url == "http://localhost:11434/v1/chat/completions"   # chat, not completions; /v1 not doubled
    assert field == "delta"
    assert len(payloads) == 2
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "coherent prompt" in payloads[0]["messages"][1]["content"]
    assert "prompt" not in payloads[0]                            # not the legacy completions shape


def test_stress_min_samples_flag_env_precedence(monkeypatch):
    """--stress-min-samples (config) > GBENCH_STRESS_MIN_SAMPLES env > class default 15,
    and the prompts-per-point floor scales down with a smaller sample floor (so points
    are faster) while the DEFAULT keeps the original 40-prompt floor unchanged."""
    monkeypatch.delenv("GBENCH_STRESS_MIN_SAMPLES", raising=False)
    r = StressTestRunner(BenchmarkConfig())
    assert r._min_steady == r.MIN_STEADY            # default 15
    assert r._min_prompts == r.MIN_PROMPTS          # default 40, unchanged

    monkeypatch.setenv("GBENCH_STRESS_MIN_SAMPLES", "5")
    r_env = StressTestRunner(BenchmarkConfig())
    assert r_env._min_steady == 5
    assert r_env._min_prompts < r_env.MIN_PROMPTS   # scaled down for a smaller floor
    assert r_env._min_prompts >= 5                  # but still enough to yield the samples

    cfg = BenchmarkConfig()
    cfg.stress_min_samples = 8                       # flag/config wins over the env above
    r_flag = StressTestRunner(cfg)
    assert r_flag._min_steady == 8


def test_lowered_min_samples_makes_a_too_few_point_pass(monkeypatch):
    """A point that meets the SLO and keeps up but has only 8 steady completions is
    TOO-FEW at the default floor (15) yet PASSES once the floor is lowered to 5."""
    def _point():
        return dict(achieved_qps=1.0, slo_attainment=1.0, p99_ttft_ms=100.0,
                    p99_tpot_ms=20.0, p99_itl_ms=20.0, p99_itl_server_ms=None,
                    mean_e2e_ms=100.0, steady_n=8, completed=8, failed=0,
                    client_bound=False, actual_send_qps=1.0, client_sched_lag_s=0.0,
                    num_client_procs=1)
    r_default = StressTestRunner(BenchmarkConfig())            # floor 15
    monkeypatch.setattr(r_default, "_measure_rate_point", lambda *a, **k: _point())
    passed_default, *_ = r_default._test_rate(1.0, None, None)
    assert passed_default is False
    assert r_default._tested_rates[1.0]["passed"] is False     # TOO-FEW (8 < 15)

    cfg = BenchmarkConfig(); cfg.stress_min_samples = 5
    r_low = StressTestRunner(cfg)                              # floor 5
    monkeypatch.setattr(r_low, "_measure_rate_point", lambda *a, **k: _point())
    passed_low, *_ = r_low._test_rate(1.0, None, None)
    assert passed_low is True                                  # 8 >= 5 -> counts


def _fake_capacity_sweep(r, monkeypatch, cap_qps):
    """Wire _test_rate so any rate <= cap_qps passes and anything above fails,
    recording the probe order. Simulates a server with a hard capacity ceiling."""
    probed = []
    def fake_test_rate(qps, model, format):
        probed.append(qps)
        passed = qps <= cap_qps
        r._tested_rates[qps] = {
            "request_rate_qps": qps, "achieved_qps": (qps if passed else 0.0),
            "passed": passed, "steady_n": (20 if passed else 0),
            "p99_ttft_ms": (100.0 if passed else 9e9),
        }
        return passed, (100.0 if passed else 9e9), 0.0, (qps if passed else 0.0)
    monkeypatch.setattr(r, "_test_rate", fake_test_rate)
    return probed


def test_sweep_is_ascending_and_stops_at_first_fail(monkeypatch):
    """The sweep must probe LOW -> HIGH and stop at the first SLO miss, so a slow
    server's sustainable rates are measured before any over-capacity probe (which
    would flood its queue and spuriously fail every later lower rate)."""
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 2500.0                       # -> cap est 0.4 -> start ~0.2
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=0.35)
    out = r._sweep_once(None, None)

    assert probed == sorted(probed)                # strictly ascending, no downward probes
    assert 0 < out["knee_qps"] <= 0.35             # knee is a sustainable rate
    assert sum(1 for q in probed if q > 0.35) <= 1 # at most ONE over-capacity probe (the boundary)
    assert max(probed) == probed[-1]               # the last probe is the highest (stopped there)


def test_sweep_ramps_down_when_start_seeded_too_high(monkeypatch):
    """If the seeded start already misses the SLO (server slower than its single-stream
    estimate), the sweep halves downward to find the small sustainable rate."""
    r = StressTestRunner(BenchmarkConfig())
    r._probe_e2e_ms = 100.0                        # -> cap est 10 -> start ~5 (too high)
    probed = _fake_capacity_sweep(r, monkeypatch, cap_qps=0.6)
    out = r._sweep_once(None, None)
    assert 0 < out["knee_qps"] <= 0.6
    assert probed[0] > 0.6                          # first probe was the (failing) high seed
    assert probed[-1] <= 0.6                        # ended on a passing lower rate
