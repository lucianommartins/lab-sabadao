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

"""Shared structural parsing/matching for function-calling evaluation suites.

Several suites were scoring tool use with free-text substring checks: "does the gold tool
name appear anywhere in the response". That credits a model for echoing a tool name out of
the prompt's own API list and ignores the arguments entirely. This module parses BOTH sides
into (name, args) and compares them structurally.

Formats handled (models and datasets each use a different one):
  * OpenAI tool-call rendering from base.py:  ``name({"a": 1}) name(a=1)``
  * bare JSON:                                ``{"name": "f", "arguments": {...}}``
  * python-ish call:                          ``f(a=1, b="x")``
  * ReAct / ToolLLaMA:                        ``Action: f`` + ``Action Input: {"a": 1}``
  * API-Bank:                                 ``API-Request: [f(a=1)]``
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

ToolCall = Tuple[str, Dict[str, Any]]

_IDENT = r"[A-Za-z_][A-Za-z0-9_.\-]*"

# OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$. Datasets that namespace their
# tools (`default_api:flight_book`) have to be rewritten before the schema is sent, so the
# same rewrite is applied to the gold before comparing.
_ILLEGAL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def _coerce(value: Any) -> Any:
    """Normalize a scalar for comparison (numbers vs strings, casing, whitespace)."""
    if isinstance(value, str):
        v = value.strip().strip("'\"").strip()
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null", ""):
            return None
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return low
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return int(f) if f.is_integer() else f
    if isinstance(value, dict):
        return {str(k).strip().lower(): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return value


def _norm_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {}
    if not isinstance(args, dict):
        return {}
    return {str(k).strip().lower(): _coerce(v) for k, v in args.items()}


def _json_objects(text: str) -> List[Dict[str, Any]]:
    """Every top-level JSON object in free text."""
    out, dec, i, n = [], json.JSONDecoder(), 0, len(text or "")
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                if isinstance(obj, dict):
                    out.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return out


def _split_top_level(s: str) -> List[str]:
    """Split on commas that are not inside quotes/brackets."""
    parts, buf, depth, in_str, quote = [], [], 0, False, ""
    for ch in s:
        if in_str:
            if ch == quote:
                in_str = False
        elif ch in "\"'":
            in_str, quote = True, ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_loose_call(snippet: str) -> Optional[ToolCall]:
    """Parse a call whose arguments are not valid Python literals.

    API-Bank golds look like ``ModifyRegistration(appointment_id=abc123,
    new_time=2023-10-10 11:00:00)`` - unquoted identifiers and bare timestamps that `ast`
    rejects outright. Treat each ``k=v`` value as a raw string.
    """
    m = re.match(rf"\s*({_IDENT})\s*\((.*)\)\s*$", snippet.strip(), re.DOTALL)
    if not m:
        return None
    name, inner = m.group(1), m.group(2)
    args: Dict[str, Any] = {}
    for idx, part in enumerate(_split_top_level(inner)):
        if "=" in part:
            k, _, v = part.partition("=")
            args[k.strip().lower()] = _coerce(v)
        else:
            args[f"__pos{idx}"] = _coerce(part)
    return name, args


def _parse_python_call(snippet: str) -> Optional[ToolCall]:
    """Parse ``f(a=1, b="x")`` (and positional args) via the AST, not regex."""
    try:
        node = ast.parse(snippet.strip(), mode="eval").body
    except Exception:
        return _parse_loose_call(snippet)
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
    if not name:
        return None
    args: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            args[str(kw.arg).lower()] = _coerce(ast.literal_eval(kw.value))
        except Exception:
            # An unquoted value (`unit=units`) parses as a Name/Attribute node, so
            # literal_eval fails while ast.parse succeeded - the loose-parser fallback
            # never runs. Dropping it to None silently discarded the argument and made
            # correct calls unscoreable; keep the source text instead.
            try:
                args[str(kw.arg).lower()] = _coerce(ast.unparse(kw.value))
            except Exception:
                args[str(kw.arg).lower()] = None
    for idx, a in enumerate(node.args):
        try:
            args[f"__pos{idx}"] = _coerce(ast.literal_eval(a))
        except Exception:
            pass
    return name, args


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Extract every tool call we can find, in order of appearance."""
    calls: List[ToolCall] = []
    if not text:
        return calls
    s = str(text)

    # 1) JSON objects carrying a name/arguments pair
    for obj in _json_objects(s):
        name = obj.get("name") or obj.get("action") or obj.get("tool") or obj.get("api")
        if not name or not isinstance(name, str):
            continue
        raw = (obj.get("arguments") if "arguments" in obj else
               obj.get("args") if "args" in obj else
               obj.get("parameters") if "parameters" in obj else
               obj.get("action_input") if "action_input" in obj else {})
        calls.append((name.strip(), _norm_args(raw)))

    # 2) ReAct / ToolLLaMA: "Action: f" with an optional "Action Input: {...}"
    # NB the separator must not be a greedy \s* - that would swallow the newline the
    # optional "Action Input" group needs, silently dropping every argument.
    for m in re.finditer(rf"Action\s*:[ \t]*({_IDENT})[ \t]*(?:\s*Action\s*Input\s*:[ \t]*(\{{.*?\}}|\S[^\n]*))?",
                         s, re.IGNORECASE | re.DOTALL):
        name = m.group(1).strip()
        if name.lower() in ("none", "finish"):
            continue
        calls.append((name, _norm_args(m.group(2) or {})))

    # 3) python-ish calls, incl. API-Bank's "API-Request: [f(a=1)]"
    for m in re.finditer(rf"\b({_IDENT})\s*\(", s):
        start = m.start()
        depth, i, in_str, quote = 0, m.end() - 1, False, ""
        while i < len(s):
            ch = s[i]
            if in_str:
                if ch == quote and s[i - 1] != "\\":
                    in_str = False
            elif ch in "\"'":
                in_str, quote = True, ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        parsed = _parse_python_call(s[start:i + 1])
        if parsed:
            calls.append(parsed)

    # de-duplicate, preserving order
    seen, uniq = set(), []
    for name, args in calls:
        key = (name.lower(), json.dumps(args, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            uniq.append((name, args))
    return uniq


def parse_gold_call(gold: Any) -> Optional[ToolCall]:
    """Parse the dataset's gold into (name, args). Handles dict/list/JSON/ReAct/API-Request."""
    if gold is None:
        return None
    if isinstance(gold, dict):
        name = gold.get("name") or gold.get("action") or gold.get("tool") or gold.get("api")
        if name:
            raw = gold.get("arguments", gold.get("args", gold.get("parameters", {})))
            return str(name).strip(), _norm_args(raw)
        return None
    if isinstance(gold, list):
        for item in gold:                      # ComplexFuncBench stores a LIST of calls
            parsed = parse_gold_call(item)
            if parsed:
                return parsed
        return None
    text = str(gold).strip()
    if not text:
        return None
    # Some datasets store the gold as a PYTHON literal, not JSON: seal_tools ships
    # "[{'api': 'analyzeEvidence', 'parameters': {...}}]". Single quotes defeat
    # json.loads, and there is no ReAct or `f(...)` form to fall back on, so the gold
    # parsed to None and every row failed regardless of the answer (seal_tools: 0/20).
    if text[:1] in "[{" and '"' not in text[:200]:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, (list, dict)):
            parsed = parse_gold_call(value)
            if parsed:
                return parsed
    # API-Bank golds are prefixed "API-Request: [f(a=1)]" while the prompt asks for "[f(...)]"
    text = re.sub(r"^\s*API-Request\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def parse_gold_calls(gold: Any) -> List[ToolCall]:
    """ALL gold calls as (name, args) - parse_gold_call() returns only the first.

    Needed by multi-call suites (seal_tools, nestful) whose gold is a LIST of calls; scoring
    only the first collapsed multi-call answers to a single call.
    """
    if gold is None:
        return []
    if isinstance(gold, dict):
        c = parse_gold_call(gold)
        return [c] if c else []
    if isinstance(gold, list):
        out: List[ToolCall] = []
        for item in gold:
            out.extend(parse_gold_calls(item))
        return out
    text = str(gold).strip()
    if not text:
        return []
    # python-literal list/dict with single quotes (seal_tools ships these): literal_eval then
    # recurse, so every call is kept rather than only calls[0].
    if text[:1] in "[{" and '"' not in text[:200]:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, (list, dict)):
            return parse_gold_calls(value)
    text = re.sub(r"^\s*API-Request\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return parse_tool_calls(text)


def gold_call_dicts(gold: Any) -> List[Dict[str, Any]]:
    """Gold as a list of ``{name, args}`` dicts, ready for score_exact_call_set/score_sequence."""
    return [{"name": n, "args": a} for n, a in parse_gold_calls(gold)]


def call_matches(gold: ToolCall, candidates: List[ToolCall], require_args: bool = True) -> bool:
    """True iff some candidate call has the gold's name and (optionally) its arguments.

    Argument matching is subset-based: every gold argument must be present with an
    equivalent value. Extra model-supplied arguments are tolerated (datasets often omit
    optional defaults), but a missing or contradicting gold argument fails.
    """
    if not gold:
        return False
    gold_name, gold_args = gold[0].strip().lower(), gold[1] or {}
    for name, args in candidates:
        if name.strip().lower() != gold_name:
            continue
        if not require_args or not gold_args:
            return True
        args = args or {}
        # Value comparison goes through _values_equal (float tolerance, case-insensitive
        # strings, and a bool-before-numeric guard so a boolean gold answered with 1 does
        # NOT match) instead of a bare `==`, which accepted True==1 and rejected 3==3.0.
        if all(k in args and _values_equal(v, args[k]) for k, v in gold_args.items()):
            return True
    return False


def score_tool_call(response_text: str, gold: Any, require_args: bool = True) -> bool:
    """Convenience: parse both sides and compare structurally."""
    g = parse_gold_call(gold)
    if not g:
        return False
    return call_matches(g, parse_tool_calls(response_text), require_args=require_args)


# --------------------------------------------------------------------------- #
# Exact FunctionCall-AST matching (`exact_tool_call_accuracy`)
#
# Distinct from call_matches() above, which is deliberately lenient: it looks for ONE gold
# call among the candidates and tolerates extra arguments. The exact match metric requires an
# exact match over the whole response - function name, argument key set, parameter types
# and values - and penalises both omitted and hallucinated calls/arguments.
# --------------------------------------------------------------------------- #

def normalize_tool_name(name: str) -> str:
    """Canonical form used to compare a gold tool name with an emitted one."""
    return _ILLEGAL_NAME_CHARS.sub("_", str(name or "").strip()).casefold()


def parse_raw_tool_calls(tool_calls: Any) -> List[ToolCall]:
    """Convert an OpenAI ``tool_calls`` array into (name, args) keeping JSON types.

    Unlike parse_tool_calls(), this does not go through the text rendering, so an INTEGER
    parameter answered with the string ``"2010"`` stays a string and can be caught by the
    type-conformance check.
    """
    out: List[ToolCall] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        out.append((str(name), dict(raw) if isinstance(raw, dict) else {}))
    return out


def _values_equal(gold: Any, pred: Any) -> bool:
    """Value equivalence: float tolerance, case-insensitive strings, structural containers.

    `bool` is checked before the numeric branch on purpose - in Python ``True == 1``, so a
    boolean parameter answered with ``1`` would otherwise be accepted.
    """
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and gold == pred
    if gold is None or pred is None:
        return gold is None and pred is None
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(gold), float(pred), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(gold, str) and isinstance(pred, (int, float)):
        return False
    if isinstance(gold, (int, float)) and isinstance(pred, str):
        try:
            return math.isclose(float(gold), float(pred.strip()), rel_tol=1e-6, abs_tol=1e-9)
        except ValueError:
            return False
    if isinstance(gold, str) and isinstance(pred, str):
        return gold.strip().casefold() == pred.strip().casefold()
    if isinstance(gold, (list, tuple)) and isinstance(pred, (list, tuple)):
        return len(gold) == len(pred) and all(_values_equal(g, p) for g, p in zip(gold, pred))
    if isinstance(gold, dict) and isinstance(pred, dict):
        gkeys = {str(k).strip().casefold() for k in gold}
        pkeys = {str(k).strip().casefold() for k in pred}
        if gkeys != pkeys:
            return False
        lowered = {str(k).strip().casefold(): v for k, v in pred.items()}
        return all(_values_equal(v, lowered[str(k).strip().casefold()]) for k, v in gold.items())
    return gold == pred


_JSON_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, (list, tuple)),
    "object": lambda v: isinstance(v, dict),
}


def _schema_properties(tool_schemas: Any, name: str) -> Dict[str, Any]:
    for tool in tool_schemas or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if normalize_tool_name(fn.get("name", "")) == normalize_tool_name(name):
            params = fn.get("parameters") or {}
            props = params.get("properties")
            return props if isinstance(props, dict) else {}
    return {}


def _types_conform(name: str, args: Dict[str, Any], props: Dict[str, Any]) -> Tuple[bool, str]:
    """Check emitted argument types against the declared schema (rubric rule 3)."""
    if not props:
        return True, ""
    for key, value in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        declared = spec.get("type")
        if value is None and spec.get("nullable"):
            continue
        check = _JSON_TYPE_CHECKS.get(declared)
        if check and not check(value):
            return False, f"{name}.{key}: expected {declared}, got {type(value).__name__}"
    return True, ""


def _call_equal(gold: Dict[str, Any], pred: ToolCall) -> bool:
    if normalize_tool_name(gold.get("name", "")) != normalize_tool_name(pred[0]):
        return False
    gargs = gold.get("args") or {}
    pargs = pred[1] or {}
    gkeys = {str(k).strip().casefold() for k in gargs}
    pkeys = {str(k).strip().casefold() for k in pargs}
    if gkeys != pkeys:                        # omitted or hallucinated argument -> fail
        return False
    lowered = {str(k).strip().casefold(): v for k, v in pargs.items()}
    return all(_values_equal(v, lowered[str(k).strip().casefold()]) for k, v in gargs.items())


def score_exact_call_set(
    gold_calls: List[Dict[str, Any]],
    pred_calls: List[ToolCall],
    tool_schemas: Any = None,
) -> Dict[str, Any]:
    """Exact match over the whole set of emitted function calls.

    Rules (docs/eval-rubrics/agent_and_tool_use_rubrics.md §1):
      1. every gold call is emitted with the same name;
      2. no missing and no hallucinated call (cardinality must match exactly, so an
         unwarranted extra call fails the sample);
      3. per call the argument key sets are equal and each value is equivalent;
      4. each emitted value conforms to the declared parameter type.

    Order is not significant: these are parallel, independent calls in one turn.
    Returns the verdict plus the sub-checks, so a run can report *why* it failed rather
    than only a pass rate.
    """
    detail: Dict[str, Any] = {
        "n_gold": len(gold_calls), "n_pred": len(pred_calls),
        "names_ok": False, "args_ok": False, "types_ok": True, "reason": "",
    }
    if not gold_calls:
        detail["reason"] = "no gold calls"
        return {"exact": False, **detail}

    gold_names = sorted(normalize_tool_name(g.get("name", "")) for g in gold_calls)
    pred_names = sorted(normalize_tool_name(p[0]) for p in pred_calls)
    detail["names_ok"] = gold_names == pred_names
    if len(pred_calls) != len(gold_calls):
        detail["reason"] = (f"expected {len(gold_calls)} call(s), got {len(pred_calls)}"
                            if pred_calls else "no tool call emitted")
        return {"exact": False, **detail}

    remaining = list(pred_calls)
    for gold in gold_calls:
        match = next((p for p in remaining if _call_equal(gold, p)), None)
        if match is None:
            same_name = [p for p in remaining
                         if normalize_tool_name(p[0]) == normalize_tool_name(gold.get("name", ""))]
            detail["reason"] = (f"arguments differ for {gold.get('name')}" if same_name
                                else f"missing call {gold.get('name')}")
            return {"exact": False, **detail}
        remaining.remove(match)
    detail["args_ok"] = True

    for name, args in pred_calls:
        # Prefer the full declared schema; fall back to the compact `param_types` the gold
        # carries, so type conformance still works when the 100+ tool schemas of the
        # request are not retained alongside the result.
        props = _schema_properties(tool_schemas, name)
        if not props:
            gold = next((g for g in gold_calls
                         if normalize_tool_name(g.get("name", "")) == normalize_tool_name(name)), None)
            props = {k: {"type": v} for k, v in ((gold or {}).get("param_types") or {}).items()}
        ok, why = _types_conform(name, args, props)
        if not ok:
            detail["types_ok"] = False
            detail["reason"] = why
            return {"exact": False, **detail}

    return {"exact": True, **detail}


def score_possible_answer(response_text: str, gold: Any,
                          calls: Optional[List[ToolCall]] = None) -> bool:
    """Score against BFCL's `possible_answer` format.

    Gold is a list of ``{func_name: {param: [accepted values...]}}``. A parameter whose
    accepted list contains "" is optional. Matching is structural: the model must emit a
    call with that function name, and every required parameter must be present with one of
    its accepted values. The previous check tested whether the function name and some value
    appeared anywhere in the response text, which a model could satisfy by echoing the
    prompt's API list without ever emitting a call.

    `calls`: pre-parsed (name, args) pairs. Pass the STRUCTURED tool calls
    (parse_raw_tool_calls(reply.tool_calls)) so scoring does not go through base.py's lossy
    text render `name({json}) name(a=1)` (which drops spaces in unquoted values, splits on
    commas inside unquoted strings, truncates namespaced function names, and swallows the
    JSON object as a positional arg). Falls back to parsing `response_text` when omitted.
    """
    if not gold:
        return False
    specs = gold if isinstance(gold, list) else [gold]
    if calls is None:
        if not response_text:
            return False
        calls = parse_tool_calls(response_text)
    if not calls:
        return False

    for spec in specs:
        if isinstance(spec, str):
            parsed = parse_gold_call(spec)
            if not parsed or not call_matches(parsed, calls):
                return False
            continue
        if not isinstance(spec, dict):
            return False
        for fname, params in spec.items():
            # EVERY candidate with this name must be tried, not just the first.
            # base.py renders one tool call twice - `f({"a": 1}) f(a=1)` - so
            # parse_tool_calls returns two entries for it: the first has the whole JSON
            # object swallowed as a positional `__pos0`, the second has the named args.
            # Binding to the first match scored correct calls as wrong (bfcl: 0/20).
            candidates = [args or {} for name, args in calls
                          if name.strip().lower() == str(fname).strip().lower()]
            if not candidates:
                return False                      # required function never called
            if not isinstance(params, dict):
                continue
            if not any(_args_satisfy(cand, params) for cand in candidates):
                return False
    return True


def _value_matches(accepted: Any, cand_val: Any) -> bool:
    """A candidate value satisfies a possible_answer accepted value if it is case-insensitively /
    structurally equal (this handles STRUCTURED args, which keep their original case and JSON
    types -- e.g. 'J.K. Rowling', ['bookA','bookB'], the string '12345'), OR equals the
    type-coerced accepted value (keeps the legacy path where a string gold like "5" is answered
    with the int 5). The text-render path relied on both sides being lowercased by _coerce; the
    structured path does not lowercase the candidate, so the case-insensitive comparator is needed."""
    return _values_equal(accepted, cand_val) or cand_val == _coerce(accepted)


def _args_satisfy(cand_args: Dict[str, Any], params: Dict[str, Any]) -> bool:
    """Does this candidate call's argument dict satisfy a BFCL `possible_answer` spec?"""
    # No UNEXPECTED parameters: possible_answer lists every acceptable param (optional ones
    # carry "" among their accepted values), so an argument key absent from the spec is a
    # hallucinated parameter and fails, per BFCL's AST checker. (A double-rendered candidate
    # carrying a positional `__pos0` fails here; the correctly-named candidate still passes
    # via score_possible_answer's any-candidate match.)
    spec_keys = {str(p).strip().lower() for p in params}
    for key in cand_args:
        if str(key).strip().lower() not in spec_keys:
            return False
    for pname, accepted in params.items():
        allowed = accepted if isinstance(accepted, list) else [accepted]
        key = str(pname).strip().lower()
        optional = any(isinstance(a, str) and a == "" for a in allowed)
        if key not in cand_args:
            # OMITTED: fine iff the param is optional ("" among its accepted values means
            # "may be omitted"); a missing required param fails.
            if optional:
                continue
            return False
        # SUPPLIED: the value must match an accepted value EVEN IF the param is optional. "" means
        # "may be omitted", NOT "any value is acceptable" - the canonical BFCL AST checker validates
        # a provided value regardless of optionality. (The old code short-circuited on any optional
        # param, over-crediting a supplied MISMATCH e.g. include_revoked=true vs accepted ["", false].)
        cand_val = cand_args[key]
        # Nested possible_answer: an accepted value can itself be a {subparam: [accepted...]}
        # object (BFCL's nested-dict params, e.g. profile_data). Recurse rather than flat-compare
        # the concrete dict against a spec whose values are accepted-lists (which never == it).
        dict_specs = [a for a in allowed if isinstance(a, dict)]
        if dict_specs and isinstance(cand_val, dict) and any(
                _args_satisfy(cand_val, ds) for ds in dict_specs):
            continue
        if any(_value_matches(a, cand_val) for a in allowed if not isinstance(a, dict)):
            continue
        return False
    return True


# --------------------------------------------------------------------------- #
# Corpus-pooled precision/recall/F1 and position-aligned sequence matching.
#
# seal_tools reports micro Tool/Parameter P/R/F1 and NESTFUL reports Full/Partial
# Sequence Match plus F1-Func / F1-Param. Both are CORPUS-level metrics: precision and
# recall pool true/false positives across the whole run, so they CANNOT be averaged per
# sample (mean-of-ratios is wrong). The scorers below return per-sample (tp, fp, fn)
# counts; a suite pools them over result['sample_traces'] in a trace-walk and calls
# prf_from_counts() once at the end.
# --------------------------------------------------------------------------- #


def _call_name_args(call: Any) -> Tuple[str, Dict[str, Any]]:
    """(name, args) from either a ToolCall tuple or a {name, args} gold dict."""
    if isinstance(call, tuple):
        return call[0], (call[1] or {})
    if isinstance(call, dict):
        name = call.get("name") or call.get("action") or call.get("tool") or call.get("api") or ""
        args = call.get("args", call.get("arguments", call.get("parameters", {}))) or {}
        return name, args
    return "", {}


def _canon_value(value: Any) -> str:
    """A hashable, comparison-stable rendering of a (coerced) argument value."""
    try:
        return json.dumps(_coerce(value), sort_keys=True, default=str)
    except Exception:
        return str(value)


def tool_items(calls: List[Any]) -> List[str]:
    """Tool-name multiset for a list of calls (F1-Func / Tool-P/R/F1)."""
    return [normalize_tool_name(_call_name_args(c)[0]) for c in (calls or [])]


def param_items(calls: List[Any]) -> List[Tuple[str, str, str]]:
    """(tool, key, value) triple multiset for parameter-level P/R/F1."""
    out: List[Tuple[str, str, str]] = []
    for c in calls or []:
        name, args = _call_name_args(c)
        nn = normalize_tool_name(name)
        for k, v in (args or {}).items():
            out.append((nn, str(k).strip().casefold(), _canon_value(v)))
    return out


def prf_counts(gold_items: List[Any], pred_items: List[Any]) -> Tuple[int, int, int]:
    """Multiset (tp, fp, fn) between two collections of hashable items."""
    g, p = Counter(gold_items), Counter(pred_items)
    tp = sum((g & p).values())
    return tp, sum((p - g).values()), sum((g - p).values())


def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Micro precision/recall/F1 from pooled counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def score_sequence(gold_calls: List[Dict[str, Any]], pred_calls: List[ToolCall]) -> Dict[str, Any]:
    """Position-aligned sequence scoring for NESTFUL-style multi-call answers.

    Returns per sample:
      * full    - the whole sequence matches in order (name+args equal, same length);
      * partial - fraction of gold positions matched at the SAME index;
      * func    - (tp, fp, fn) over the tool-name multiset (pool -> F1-Func);
      * param   - (tp, fp, fn) over (tool, key, value) triples (pool -> F1-Param).

    Order matters (a nested call consumes an earlier call's output), so matching is
    index-aligned, unlike the order-insensitive set membership of call_matches().
    """
    n, m = len(gold_calls), len(pred_calls)
    pos_correct = sum(1 for i in range(min(n, m)) if _call_equal(gold_calls[i], pred_calls[i]))
    full = (n == m and pos_correct == n)
    partial = (pos_correct / n) if n else (1.0 if m == 0 else 0.0)
    return {
        "full": full,
        "partial": partial,
        "func": prf_counts(tool_items(gold_calls), tool_items(pred_calls)),
        "param": prf_counts(param_items(gold_calls), param_items(pred_calls)),
    }
