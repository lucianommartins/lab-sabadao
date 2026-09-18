# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""WS8: the analytics endpoints surface performance (latency/throughput), not only quality pass rate."""

from fastapi.testclient import TestClient

import service.routes.analytics as A
from service.main import app
from service.routes.analytics import _num, _perf_metrics

client = TestClient(app)


def test_num_rejects_bool_and_nonnumeric():
    assert _num(1.5) == 1.5 and _num(3) == 3
    assert _num(True) is None and _num("x") is None and _num(None) is None


def test_perf_metrics_extracts_serving_and_throughput():
    results = {
        "serving": {"ttft_p50_ms": 42.0, "tpot_p50_ms": 8.5, "output_throughput": 1234.0,
                    "request_throughput": 12.3, "unrelated": "x"},
        "throughput": {"output_throughput": 5000.0, "total_token_throughput": 6000.0},
        "quality": {"pass_rate": 90.0},
    }
    p = _perf_metrics(results)
    assert p["ttft_p50_ms"] == 42.0 and p["tpot_p50_ms"] == 8.5
    assert p["output_throughput"] == 1234.0 and p["request_throughput"] == 12.3
    assert p["throughput_output_tok_s"] == 5000.0 and p["throughput_total_tok_s"] == 6000.0
    assert "unrelated" not in p


def test_perf_metrics_empty_without_perf_pillars():
    assert _perf_metrics({"quality": {"pass_rate": 90.0}}) == {}
    assert _perf_metrics({}) == {}


def test_perf_metrics_ignores_nonnumeric_and_bool():
    assert _perf_metrics({"serving": {"ttft_p50_ms": None, "tpot_p50_ms": "n/a",
                                      "output_throughput": True}}) == {}


def test_run_analytics_endpoint_includes_performance(monkeypatch):
    fake = {"status": "completed", "tags": [], "models": ["m"], "results": {
        "quality": {"pass_rate": 80.0, "total_scenarios": 10, "passed_scenarios": 8,
                    "raw_results": {"scenarios": []}},
        "serving": {"ttft_p50_ms": 50.0, "output_throughput": 900.0}}}
    monkeypatch.setattr(A, "get_run_details", lambda rid: fake)
    r = client.get("/api/analytics/run123")
    assert r.status_code == 200
    body = r.json()
    assert body["performance"]["ttft_p50_ms"] == 50.0
    assert body["metrics"]["ttft_p50_ms"] == 50.0        # perf merged into metrics
    assert body["metrics"]["pass_rate"] == 0.8


def test_performance_only_run_still_surfaces_metrics(monkeypatch):
    fake = {"status": "completed", "tags": [], "models": ["m"], "results": {
        "serving": {"tpot_p50_ms": 7.0, "output_throughput": 1500.0}}}   # no quality pillar
    monkeypatch.setattr(A, "get_run_details", lambda rid: fake)
    r = client.get("/api/analytics/perfonly")
    assert r.status_code == 200
    body = r.json()
    assert body["performance"]["tpot_p50_ms"] == 7.0
    assert body["metrics"]["output_throughput"] == 1500.0    # not an empty payload anymore
