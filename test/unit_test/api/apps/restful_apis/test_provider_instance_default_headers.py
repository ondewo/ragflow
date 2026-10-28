#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Tests that the provider instance endpoints carry `default_headers` through.

A model served behind a gateway needs custom HTTP headers on every request, and the
model clients read them from the instance `extra` blob. The instance create and update
endpoints are the only way to set them, so an omitted forward would leave the field
silently unsettable while the request still succeeds.

`provider_api.py` is a route module: it expects the `manager` blueprint the app loader
injects as a module global, so the module namespace is seeded with a fake before the
body runs, and the handlers are then called directly.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.p2


class _FakeRequest:
    """The one piece of quart the handlers touch."""

    def __init__(self):
        self.json_body = {}

    async def get_json(self):
        return self.json_body


def _identity_decorator(func):
    return func


def _stub(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, _, child_name = name.rpartition(".")
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None:
            monkeypatch.setattr(parent_module, child_name, module, raising=False)
    return module


def _load_routes(monkeypatch):
    """Load provider_api.py against a fake blueprint and a fake service layer.

    Returns the module, the fake request whose body each test sets, and the recorded
    keyword arguments of every service call.
    """
    fake_request = _FakeRequest()
    calls: dict[str, dict] = {}

    async def _create_provider_instance(*args, **kwargs):
        calls["create"] = {"args": args, "kwargs": kwargs}
        return True, "success"

    async def _update_provider_instance(*args, **kwargs):
        calls["update"] = {"args": args, "kwargs": kwargs}
        return True, "success"

    _stub(monkeypatch, "quart", request=fake_request, Response=object)
    _stub(monkeypatch, "api.apps", login_required=_identity_decorator, current_user=SimpleNamespace(id="user-1"))
    _stub(
        monkeypatch,
        "api.utils.api_utils",
        add_tenant_id_to_kwargs=_identity_decorator,
        get_error_argument_result=lambda message="": {"error": message},
        get_error_data_result=lambda message="": {"error": message},
        get_result=lambda **kwargs: {"ok": kwargs},
    )
    _stub(
        monkeypatch,
        "api.apps.services",
        provider_api_service=SimpleNamespace(
            create_provider_instance=_create_provider_instance,
            update_provider_instance=_update_provider_instance,
        ),
    )

    module_path = Path(__file__).resolve().parents[5] / "api" / "apps" / "restful_apis" / "provider_api.py"
    spec = importlib.util.spec_from_file_location("provider_api_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    # The app loader hands every route module its blueprint as a bare global.
    module.__dict__["manager"] = SimpleNamespace(route=lambda *args, **kwargs: _identity_decorator)
    monkeypatch.setitem(sys.modules, "provider_api_under_test", module)
    spec.loader.exec_module(module)
    return module, fake_request, calls


async def test_create_instance_forwards_default_headers(monkeypatch):
    module, fake_request, calls = _load_routes(monkeypatch)
    fake_request.json_body = {
        "instance_name": "primary",
        "api_key": "sk-new",
        "base_url": "http://vllm.internal/v1",
        "model_info": [],
        "default_headers": {"X-Tenant": "oeamtc"},
    }

    await module.create_provider_instance(tenant_id="tenant-1", provider_id_or_name="VLLM")

    assert calls["create"]["kwargs"]["default_headers"] == {"X-Tenant": "oeamtc"}


async def test_update_instance_forwards_default_headers(monkeypatch):
    module, fake_request, calls = _load_routes(monkeypatch)
    fake_request.json_body = {
        "instance_name": "primary",
        "api_key": "sk-new",
        "base_url": "http://vllm.internal/v1",
        "model_info": [],
        "verify": False,
        "default_headers": {"X-Tenant": "oeamtc"},
    }

    await module.update_provider_instance(tenant_id="tenant-1", provider_id_or_name="VLLM", instance_id_or_name="instance-1")

    assert calls["update"]["kwargs"]["default_headers"] == {"X-Tenant": "oeamtc"}


async def test_update_instance_omitting_default_headers_leaves_them_unset(monkeypatch):
    """An omitted field must reach the service as None, which keeps the stored headers."""
    module, fake_request, calls = _load_routes(monkeypatch)
    fake_request.json_body = {
        "instance_name": "primary",
        "api_key": "sk-new",
        "base_url": "http://vllm.internal/v1",
        "model_info": [],
    }

    await module.update_provider_instance(tenant_id="tenant-1", provider_id_or_name="VLLM", instance_id_or_name="instance-1")

    assert calls["update"]["kwargs"]["default_headers"] is None
