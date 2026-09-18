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
"""Regression: MultiPL-E code assembly for CHAT completions.

WS10 audit found signature reformatting (js 'foo(nums) {' vs prompt 'foo(nums){') broke the dedup and
emitted the signature twice. A first brace-rebalance fix then over-corrected: it stripped the model's
closing '}' assuming the tests close the prompt's scope, which is TRUE for scala but FALSE for js
(whose tests are self-contained 'const assert=...'), turning a correct js solution into a SyntaxError.
The fix is DATA-DRIVEN: balance the model's braces against the ACTUAL test suite so `code + tests` is
brace-balanced for every language."""

from gbench.runners.eval_suites.multipl_e import (
    _assemble_code, _balance_against_tests, _find_decl_ws_insensitive,
)

_net = lambda s: s.count("{") - s.count("}")


def test_declaration_match_is_whitespace_insensitive():
    code = "function foo(nums) {\n  return 1;\n}"
    assert _find_decl_ws_insensitive(code, "function foo(nums){") == 0   # prompt had no space


def test_signature_not_emitted_twice_when_reformatted():
    prompt = "function foo(nums){\n"
    resp = "```js\nfunction foo(nums) {\n  return nums.length;\n}\n```"
    asm = _assemble_code(resp, prompt, [])
    assert asm.count("function foo") == 1


def test_self_contained_tests_keep_the_functions_close():
    # JS tests close nothing (self-contained), so the model's complete function must keep its '}'.
    code = "function foo(nums) {\n  return nums.length;\n}"
    tests = "const assert = require('assert');\nassert.equal(foo([1,2]), 2);"
    out = _balance_against_tests(code, tests)
    assert out == code                        # nothing stripped
    assert _net(out + "\n" + tests) == 0      # program brace-balanced


def test_scope_closing_tests_strip_the_models_extra_closers():
    # Scala tests close the prompt's object+def scopes, so the model's 2 closers are stripped.
    code = "object P {\n  def foo(): Int = {\n    42\n  }\n}"
    tests = "  test(\"t\") { assert(foo() == 42) }\n}\n}"   # net -2 (closes def + object)
    out = _balance_against_tests(code, tests)
    assert _net(out + "\n" + tests) == 0      # program brace-balanced
    assert out.count("}") < code.count("}")   # excess closers removed


def test_indentation_language_unaffected():
    code = "def foo(n):\n    return n + 1"
    assert _balance_against_tests(code, "assert foo(1) == 2") == code   # no braces -> no-op


def test_never_adds_braces_for_underclosed_code():
    # An under-closed program is a genuine model error; balancing must NOT invent closers.
    code = "function foo() {\n  return 1;"        # missing '}'
    tests = "assert(foo() === 1);"
    assert _balance_against_tests(code, tests) == code.rstrip("\n")
