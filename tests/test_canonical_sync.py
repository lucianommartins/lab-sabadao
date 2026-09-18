# -*- coding: utf-8 -*-
"""Tests for the canonical-sync provenance registry (WS1)."""

from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC, canonical_sync_for
from gbench.runners.eval_suites import SUITES


def test_known_suite_is_reconciled_with_required_fields():
    cs = canonical_sync_for("gaia2")
    assert cs["status"] == "reconciled"
    assert cs["synced"] == "2026-09-09"
    assert cs["upstream"] and cs["method"]


def test_unknown_suite_is_honestly_unverified():
    cs = canonical_sync_for("definitely_not_a_suite")
    assert cs["status"] == "unverified"
    assert "not yet" in cs["note"]


def test_every_entry_has_a_date_and_method():
    for name, entry in CANONICAL_SYNC.items():
        assert entry.get("synced"), f"{name} missing synced date"
        assert entry.get("method"), f"{name} missing method"


def test_no_stale_registry_keys():
    # Every provenance entry must name a currently-registered suite (catch typos / removed suites).
    unknown = sorted(k for k in CANONICAL_SYNC if k not in SUITES)
    assert not unknown, f"canonical_sync.py names suites not in SUITES: {unknown}"


def test_every_registered_suite_is_reconciled():
    # WS1 acceptance: every built-in suite carries a provenance stamp. A newly added suite must
    # get a canonical_sync entry (do not invent a date/commit; reconcile it first) before it ships.
    missing = sorted(s for s in SUITES if s not in CANONICAL_SYNC)
    assert not missing, f"suites with no canonical_sync entry (reconcile then stamp): {missing}"


def test_stamp_shape_is_json_safe():
    # The stamp is attached to every result and serialized, so it must be plain dict/str.
    cs = canonical_sync_for("wildclawbench")
    assert isinstance(cs, dict) and all(isinstance(v, str) for v in cs.values())
