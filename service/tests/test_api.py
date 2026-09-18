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

from fastapi.testclient import TestClient
from service.main import app
from service.config import settings

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "environment": settings.GBENCH_ENV}

def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_list_runs():
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_run_not_found():
    response = client.get("/api/runs/non_existent_run")
    assert response.status_code == 404
