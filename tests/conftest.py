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

"""Shared pytest configuration.

gbench defaults to `GBENCH_SANDBOX=required`: suites that execute model-written code refuse
to run without bubblewrap isolation. The unit tests run in a trusted CI environment that
does not ship bubblewrap and exercise scoring logic directly (e.g. codeforces/lcb/scicode
graders), so default the suite to unsandboxed execution here. Individual sandbox tests
still `monkeypatch.setenv("GBENCH_SANDBOX", ...)` to assert the required/bwrap/none
behaviour; `setdefault` leaves any explicit operator/CI value (e.g. a CI that installs
bwrap and sets `bwrap`) untouched.
"""

import os

os.environ.setdefault("GBENCH_SANDBOX", "none")
