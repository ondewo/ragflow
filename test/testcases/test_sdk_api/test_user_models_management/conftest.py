from typing import Any, Dict, List

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


def add_model(client: RAGFlow, **kwargs: Any) -> Any:
    """Helper function to add models via API"""
    res: requests.Response = client.post("/models", kwargs)
    res_json: Dict[str, Any] = res.json()
    if res_json.get("code") != 0:
        raise Exception(res_json.get("message"))
    return res_json.get("data")


def remove_model(client: RAGFlow, **kwargs: Any) -> Any:
    """Helper function to remove models via API"""
    res: requests.Response = client.delete("/models", kwargs)
    res_json: Dict[str, Any] = res.json()
    if res_json.get("code") != 0:
        raise Exception(res_json.get("message"))
    return res_json.get("data")


@pytest.fixture(scope="class")
def cleanup_added_models(request: FixtureRequest, client: RAGFlow) -> None:
    """Fixture to clean up models added during tests (for TestAddModel class)"""
    # Track factories that might be added during tests
    factories_to_cleanup: List[str] = ["Builtin", "LocalAI", "Ollama", "Xinference", "LM-Studio", "GPUStack", "FastEmbed"]
    
    def cleanup() -> None:
        # Remove test factories that were added during tests
        for factory in factories_to_cleanup:
            try:
                remove_model(client, llm_factory=factory)
            except Exception:
                # Ignore errors during cleanup (factory might not exist)
                pass
    
    request.addfinalizer(cleanup)