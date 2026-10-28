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
"""
The custom per-model HTTP headers have to survive the trip from storage to client.

A model instance stores them in its ``extra`` JSON blob, and
``TenantLLMService.model_instance`` reads them off the resolved ``model_config``.
The two model_config producers in this module are the only join between those
two ends, so a header that does not appear here never reaches the HTTP layer and
the whole feature is silently inert.
"""

from types import SimpleNamespace

import pytest

from common.constants import ActiveStatusEnum
from api.db.joint_services import tenant_model_service as tms

GATEWAY_HEADERS = {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}


def _install_provider_instance_lookups(monkeypatch, instance_extra: str):
    """Point the three service lookups at one in-memory provider/instance/model triple."""
    provider = SimpleNamespace(id="provider-1", provider_name="VLLM")
    instance = SimpleNamespace(id="instance-1", api_key="sk-test", extra=instance_extra)
    model = SimpleNamespace(
        model_name="qwen-chat",
        model_type="chat",
        status=ActiveStatusEnum.ACTIVE.value,
        extra="{}",
    )

    monkeypatch.setattr(
        tms.TenantModelProviderService,
        "get_by_tenant_id_and_provider_name",
        lambda tenant_id, provider_name: provider,
    )
    monkeypatch.setattr(
        tms.TenantModelInstanceService,
        "get_by_provider_id_and_instance_name",
        lambda provider_id, instance_name: instance,
    )
    monkeypatch.setattr(
        tms.TenantModelService,
        "get_by_provider_id_and_instance_id_and_model_type_and_model_name",
        lambda provider_id, instance_id, model_type, model_name: model,
    )
    monkeypatch.setattr(tms.settings, "FACTORY_LLM_INFOS", [])
    return provider, instance, model


@pytest.mark.p1
def test_provider_instance_config_carries_the_configured_headers(monkeypatch):
    _install_provider_instance_lookups(monkeypatch, '{"base_url": "https://vllm.internal/v1", "default_headers": {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}}')

    config = tms.get_model_config_from_provider_instance("tenant-1", "chat", "qwen-chat@default@VLLM")

    assert config["default_headers"] == GATEWAY_HEADERS


@pytest.mark.p1
@pytest.mark.parametrize(
    "instance_extra",
    [
        pytest.param("{}", id="no-extra-fields"),
        pytest.param('{"base_url": "https://vllm.internal/v1"}', id="other-fields-only"),
        pytest.param('{"default_headers": {}}', id="empty-mapping"),
    ],
)
def test_provider_instance_config_omits_the_key_when_no_headers_are_configured(monkeypatch, instance_extra: str):
    """An instance without headers keeps the model_config shape every provider class is built from."""
    _install_provider_instance_lookups(monkeypatch, instance_extra)

    config = tms.get_model_config_from_provider_instance("tenant-1", "chat", "qwen-chat@default@VLLM")

    assert "default_headers" not in config


def _install_by_id_lookups(monkeypatch, instance_extra: str):
    """Point get_model_config_by_id's lookups at one in-memory triple owned by the calling tenant."""
    provider = SimpleNamespace(id="provider-1", provider_name="VLLM", tenant_id="tenant-1")
    instance = SimpleNamespace(id="instance-1", api_key="sk-test", extra=instance_extra)
    model = SimpleNamespace(
        id="model-1",
        model_name="qwen-chat",
        model_type=tms.calculate_model_type("chat"),
        provider_id="provider-1",
        instance_id="instance-1",
        status=ActiveStatusEnum.ACTIVE.value,
        extra="{}",
    )

    monkeypatch.setattr(tms.TenantModelService, "get_by_id", lambda model_id: (True, model))
    monkeypatch.setattr(tms.TenantModelProviderService, "get_by_id", lambda provider_id: (True, provider))
    monkeypatch.setattr(tms.TenantModelInstanceService, "get_by_id", lambda instance_id: (True, instance))
    return provider, instance, model


@pytest.mark.p1
def test_by_id_config_carries_the_configured_headers(monkeypatch):
    _install_by_id_lookups(monkeypatch, '{"default_headers": {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}}')

    config = tms.get_model_config_by_id("tenant-1", "chat", "model-1")

    assert config["default_headers"] == GATEWAY_HEADERS


@pytest.mark.p1
def test_by_id_config_omits_the_key_when_no_headers_are_configured(monkeypatch):
    _install_by_id_lookups(monkeypatch, '{"base_url": "https://vllm.internal/v1"}')

    config = tms.get_model_config_by_id("tenant-1", "chat", "model-1")

    assert "default_headers" not in config
