from typing import Any, Dict

import pytest
import requests
from pytest import FixtureRequest
from ragflow_sdk import RAGFlow


def set_default_models(client: RAGFlow, **kwargs: Any) -> Any:
    """Helper function to set default models via API"""
    res: requests.Response = client.post("/models/default", kwargs)
    res_json: Dict[str, Any] = res.json()
    if res_json.get("code") != 0:
        raise Exception(res_json.get("message"))
    return res_json.get("data")


def get_default_models(client: RAGFlow) -> Dict[str, Any]:
    """Helper function to get default models via API"""
    res: requests.Response = client.get("/models/default")
    res_json: Dict[str, Any] = res.json()
    if res_json.get("code") != 0:
        raise Exception(res_json.get("message"))
    return res_json.get("data")


def list_user_models(client: RAGFlow, include_details: bool = False) -> Dict[str, Any]:
    """Helper function to list user models via API"""
    params: Dict[str, str] = {"include_details": "true"} if include_details else {}
    res: requests.Response = client.get("/models", params=params)
    res_json: Dict[str, Any] = res.json()
    if res_json.get("code") != 0:
        raise Exception(res_json.get("message"))
    return res_json.get("data")


@pytest.fixture(scope="class", autouse=True)
def restore_default_models(request: FixtureRequest, client: RAGFlow) -> Dict[str, Any]:
    """Fixture to save and restore default models before/after test class"""
    # Save initial state before tests
    initial_models: Dict[str, Any] = get_default_models(client)
    
    def cleanup() -> None:
        # Restore initial state after all tests in the class
        try:
            # Only restore non-empty values to avoid "at least one model ID" error
            restore_payload: Dict[str, str] = {}
            for key, value in initial_models.items():
                if value:  # Only include non-empty values
                    restore_payload[key] = value
            
            if restore_payload:
                set_default_models(client, **restore_payload)
        except Exception:
            # Ignore errors during cleanup
            pass
    
    request.addfinalizer(cleanup)
    return initial_models