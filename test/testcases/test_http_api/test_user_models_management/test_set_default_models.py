#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
from typing import Any, Dict

import pytest
from common import add_model, get_default_models, list_user_models, set_default_models
from common.constants import RetCode
from configs import INVALID_API_TOKEN
from libs.auth import RAGFlowHttpApiAuth


@pytest.mark.p1
class TestAuthorization:
    @pytest.mark.parametrize(
        "invalid_auth, expected_code, expected_message",
        [
            (None, RetCode.SUCCESS, "`Authorization` can't be empty"),
            (
                RAGFlowHttpApiAuth(INVALID_API_TOKEN),
                RetCode.AUTHENTICATION_ERROR,
                "Authentication error: API key is invalid!",
            ),
        ],
        ids=["empty_auth", "invalid_api_token"],
    )
    def test_invalid_auth(self, invalid_auth, expected_code, expected_message):
        res = set_default_models(invalid_auth, {"llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestSetDefaultModelsBuiltin:
    """Test setting built-in default models"""

    @pytest.mark.p1
    def test_set_builtin_model_exists(self, HttpApiAuth):
        """Test setting a builtin model that exists in the system"""
        # Builtin models are always allowed, but we need to use a model that actually exists
        # Try common builtin models - at least one should exist
        builtin_models = [
            "BAAI/bge-small-en-v1.5@Builtin",  # Common embedding model
            "BAAI/bge-m3@Builtin",  # Another embedding model
            "Qwen/Qwen3-Embedding-0.6B@Builtin",  # Another embedding model
        ]
        
        for model_id in builtin_models:
            res = set_default_models(HttpApiAuth, {"embd_id": model_id})
            if res["code"] == RetCode.SUCCESS:
                # Model exists, verify it was set
                get_res = get_default_models(HttpApiAuth)
                assert get_res["code"] == RetCode.SUCCESS, get_res
                assert get_res["data"]["embd_id"] == model_id
                return
        
        # If none worked, skip the test (no builtin models available)
        pytest.skip("No builtin models available in this RAGFlow instance")

    @pytest.mark.p1
    def test_set_builtin_model_not_exists(self, HttpApiAuth):
        """Test setting a builtin model that doesn't exist in the system"""
        # Use a model name that definitely doesn't exist
        model_id = "nonexistent-builtin-model@Builtin"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        # Even though it's builtin, if the model doesn't exist, it should fail
        # The API checks if model exists in TenantLLMService, and builtin models
        # are only allowed if they exist in the system
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == f"Model '{model_id}' (type: chat) is not configured. Please add the model first using POST /api/v1/models", res

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "model_type, model_field",
        [
            ("chat", "llm_id"),
            ("embedding", "embd_id"),
            ("image2text", "img2txt_id"),
            ("speech2text", "asr_id"),
            ("rerank", "rerank_id"),
            ("tts", "tts_id"),
        ],
        ids=["chat", "embedding", "image2text", "speech2text", "rerank", "tts"],
    )
    def test_set_builtin_model_by_type(self, HttpApiAuth, model_type, model_field):
        """Test setting builtin models for each model type"""
        # Try to find an actual builtin model of this type
        # First, get available models to see what exists
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] == RetCode.SUCCESS and "Builtin" in models_res["data"]:
            builtin_llm = models_res["data"]["Builtin"]["llm"]
            # Find a model of the requested type
            for model in builtin_llm:
                if model.get("type") == model_type:
                    model_id = f"{model['name']}@Builtin"
                    payload = {model_field: model_id}
                    res = set_default_models(HttpApiAuth, payload)
                    assert res["code"] == RetCode.SUCCESS, res
                    # Verify it was set
                    get_res = get_default_models(HttpApiAuth)
                    assert get_res["code"] == RetCode.SUCCESS, get_res
                    assert get_res["data"][model_field] == model_id
                    return
        
        # If no builtin model of this type exists, skip
        pytest.skip(f"No builtin {model_type} models available in this RAGFlow instance")

    @pytest.mark.p2
    def test_set_multiple_builtin_models_all_valid(self, HttpApiAuth):
        """Test setting multiple builtin models when all are valid"""
        # Get available builtin models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS or "Builtin" not in models_res["data"]:
            pytest.skip("No builtin models available")
        
        builtin_llm = models_res["data"]["Builtin"]["llm"]
        payload = {}
        
        # Find models of different types
        for model in builtin_llm:
            model_type = model.get("type")
            model_id = f"{model['name']}@Builtin"
            if model_type == "chat" and "llm_id" not in payload:
                payload["llm_id"] = model_id
            elif model_type == "embedding" and "embd_id" not in payload:
                payload["embd_id"] = model_id
            elif model_type == "image2text" and "img2txt_id" not in payload:
                payload["img2txt_id"] = model_id
        
        if len(payload) < 2:
            pytest.skip("Not enough builtin model types available")
        
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify all were set
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        for field, model_id in payload.items():
            assert get_res["data"][field] == model_id

    @pytest.mark.p2
    def test_set_multiple_builtin_models_mix_valid_invalid(self, HttpApiAuth):
        """Test setting multiple builtin models with mix of valid and invalid"""
        # Get one valid builtin model
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS or "Builtin" not in models_res["data"]:
            pytest.skip("No builtin models available")
        
        builtin_llm = models_res["data"]["Builtin"]["llm"]
        if not builtin_llm:
            pytest.skip("No builtin models available")
        
        # Use first available model as valid
        valid_model = f"{builtin_llm[0]['name']}@Builtin"
        model_type = builtin_llm[0].get("type")
        field_map = {
            "chat": "llm_id",
            "embedding": "embd_id",
            "image2text": "img2txt_id",
            "speech2text": "asr_id",
            "rerank": "rerank_id",
            "tts": "tts_id",
        }
        valid_field = field_map.get(model_type, "llm_id")
        
        # Mix valid and invalid
        payload = {
            valid_field: valid_model,
            "llm_id": "nonexistent-model@Builtin",  # Invalid
        }
        
        res = set_default_models(HttpApiAuth, payload)
        # Should fail because one model is invalid
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "not configured" in res["message"], res

    @pytest.mark.p2
    def test_set_multiple_builtin_models_all_invalid(self, HttpApiAuth):
        """Test setting multiple builtin models when all are invalid"""
        payload = {
            "llm_id": "nonexistent-model1@Builtin",
            "embd_id": "nonexistent-model2@Builtin",
        }
        res = set_default_models(HttpApiAuth, payload)
        # Should fail with error about first invalid model
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "not configured" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestSetDefaultModelsConfigured:
    """Test setting configured (non-builtin) default models"""

    @pytest.mark.p1
    def test_set_configured_model_exists(self, HttpApiAuth):
        """Test setting a model that exists and is configured for the tenant"""
        # First, try to add a factory (e.g., LocalAI which skips validation)
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] == RetCode.SUCCESS:
            # Get the models that were added
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS and "LocalAI" in models_res["data"]:
                localai_llm = models_res["data"]["LocalAI"]["llm"]
                if localai_llm:
                    # Use first available model
                    model = localai_llm[0]
                    model_id = f"{model['name']}@LocalAI"
                    model_type = model.get("type")
                    field_map = {
                        "chat": "llm_id",
                        "embedding": "embd_id",
                        "image2text": "img2txt_id",
                        "speech2text": "asr_id",
                        "rerank": "rerank_id",
                        "tts": "tts_id",
                    }
                    field = field_map.get(model_type, "llm_id")
                    
                    # Set it as default
                    res = set_default_models(HttpApiAuth, {field: model_id})
                    assert res["code"] == RetCode.SUCCESS, res
                    
                    # Verify it was set
                    get_res = get_default_models(HttpApiAuth)
                    assert get_res["code"] == RetCode.SUCCESS, get_res
                    assert get_res["data"][field] == model_id
                    return
        
        pytest.skip("No configured models available for testing")

    @pytest.mark.p1
    def test_set_configured_model_not_configured(self, HttpApiAuth):
        """Test setting a model that exists but is not configured for the tenant"""
        # Use a model from a factory that exists but hasn't been added
        # OpenAI is a common factory, but we won't add it
        model_id = "gpt-4@OpenAI"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        # Should fail because model is not configured
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == f"Model '{model_id}' (type: chat) is not configured. Please add the model first using POST /api/v1/models", res

    @pytest.mark.p1
    def test_set_configured_model_not_exists(self, HttpApiAuth):
        """Test setting a model that does not exist"""
        model_id = "nonexistent-model@OpenAI"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        # Should fail because model doesn't exist
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == f"Model '{model_id}' (type: chat) is not configured. Please add the model first using POST /api/v1/models", res

    @pytest.mark.p2
    def test_set_multiple_configured_models_all_valid(self, HttpApiAuth):
        """Test setting multiple configured models when all are valid"""
        # Add a factory first
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS or "LocalAI" not in models_res["data"]:
            pytest.skip("LocalAI models not available")
        
        builtin_llm = models_res["data"]["LocalAI"]["llm"]
        payload = {}
        
        # Find models of different types
        for model in builtin_llm:
            model_type = model.get("type")
            model_id = f"{model['name']}@LocalAI"
            if model_type == "chat" and "llm_id" not in payload:
                payload["llm_id"] = model_id
            elif model_type == "embedding" and "embd_id" not in payload:
                payload["embd_id"] = model_id
        
        if len(payload) < 2:
            pytest.skip("Not enough LocalAI model types available")
        
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify all were set
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        for field, model_id in payload.items():
            assert get_res["data"][field] == model_id

    @pytest.mark.p2
    def test_set_multiple_configured_models_mix_valid_invalid(self, HttpApiAuth):
        """Test setting multiple configured models with mix of valid and invalid"""
        # Add a factory first
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS or "LocalAI" not in models_res["data"]:
            pytest.skip("LocalAI models not available")
        
        if not models_res["data"]["LocalAI"]["llm"]:
            pytest.skip("No LocalAI models available")
        
        # Use first available model as valid
        valid_model = f"{models_res['data']['LocalAI']['llm'][0]['name']}@LocalAI"
        
        # Mix valid and invalid
        payload = {
            "llm_id": valid_model,
            "embd_id": "nonexistent-model@OpenAI",  # Invalid - not configured
        }
        
        res = set_default_models(HttpApiAuth, payload)
        # Should fail because one model is invalid
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "not configured" in res["message"], res

    @pytest.mark.p2
    def test_set_multiple_configured_models_all_invalid(self, HttpApiAuth):
        """Test setting multiple configured models when all are invalid"""
        payload = {
            "llm_id": "nonexistent-model1@OpenAI",
            "embd_id": "nonexistent-model2@Anthropic",
        }
        res = set_default_models(HttpApiAuth, payload)
        # Should fail with error about first invalid model
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "not configured" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestSetDefaultModelsClearing:
    """Test clearing default models"""

    @pytest.mark.p1
    def test_clear_one_model_with_empty_string(self, HttpApiAuth):
        """Test that setting a model to empty string clears it"""
        # First set a model
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] == RetCode.SUCCESS and "Builtin" in models_res["data"]:
            builtin_llm = models_res["data"]["Builtin"]["llm"]
            if builtin_llm:
                model = builtin_llm[0]
                model_id = f"{model['name']}@Builtin"
                model_type = model.get("type")
                field_map = {
                    "chat": "llm_id",
                    "embedding": "embd_id",
                    "image2text": "img2txt_id",
                    "speech2text": "asr_id",
                    "rerank": "rerank_id",
                    "tts": "tts_id",
                }
                field = field_map.get(model_type, "llm_id")
                
                # Set it
                res = set_default_models(HttpApiAuth, {field: model_id})
                if res["code"] == RetCode.SUCCESS:
                    # Now clear it with empty string (need at least one non-empty field)
                    # Set another field to satisfy "at least one" requirement
                    other_field = "embd_id" if field != "embd_id" else "llm_id"
                    res = set_default_models(HttpApiAuth, {field: "", other_field: model_id})
                    assert res["code"] == RetCode.SUCCESS, res
                    
                    # Verify it was cleared
                    get_res = get_default_models(HttpApiAuth)
                    assert get_res["code"] == RetCode.SUCCESS, get_res
                    assert get_res["data"][field] == ""
                    return
        
        pytest.skip("No models available for clearing test")

    @pytest.mark.p1
    def test_clear_one_model_with_whitespace(self, HttpApiAuth):
        """Test that setting a model to whitespace clears it (whitespace is processed as empty)"""
        # First set a model
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] == RetCode.SUCCESS and "Builtin" in models_res["data"]:
            builtin_llm = models_res["data"]["Builtin"]["llm"]
            if builtin_llm:
                model = builtin_llm[0]
                model_id = f"{model['name']}@Builtin"
                model_type = model.get("type")
                field_map = {
                    "chat": "llm_id",
                    "embedding": "embd_id",
                    "image2text": "img2txt_id",
                    "speech2text": "asr_id",
                    "rerank": "rerank_id",
                    "tts": "tts_id",
                }
                field = field_map.get(model_type, "llm_id")
                
                # Set it
                res = set_default_models(HttpApiAuth, {field: model_id})
                if res["code"] == RetCode.SUCCESS:
                    # Clear with whitespace (need at least one non-empty field)
                    other_field = "embd_id" if field != "embd_id" else "llm_id"
                    res = set_default_models(HttpApiAuth, {field: "   ", other_field: model_id})
                    assert res["code"] == RetCode.SUCCESS, res
                    
                    # Verify it was cleared (whitespace becomes empty string)
                    get_res = get_default_models(HttpApiAuth)
                    assert get_res["code"] == RetCode.SUCCESS, get_res
                    assert get_res["data"][field] == ""
                    return
        
        pytest.skip("No models available for clearing test")

    @pytest.mark.p2
    def test_clear_multiple_models(self, HttpApiAuth):
        """Test clearing multiple models simultaneously"""
        # First set multiple models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] == RetCode.SUCCESS and "Builtin" in models_res["data"]:
            builtin_llm = models_res["data"]["Builtin"]["llm"]
            if len(builtin_llm) >= 2:
                # Set two models
                model1 = builtin_llm[0]
                model2 = builtin_llm[1] if len(builtin_llm) > 1 else builtin_llm[0]
                model1_id = f"{model1['name']}@Builtin"
                model2_id = f"{model2['name']}@Builtin"
                
                model1_type = model1.get("type")
                model2_type = model2.get("type")
                field_map = {
                    "chat": "llm_id",
                    "embedding": "embd_id",
                    "image2text": "img2txt_id",
                    "speech2text": "asr_id",
                    "rerank": "rerank_id",
                    "tts": "tts_id",
                }
                field1 = field_map.get(model1_type, "llm_id")
                field2 = field_map.get(model2_type, "embd_id")
                
                # Set both
                res = set_default_models(HttpApiAuth, {field1: model1_id, field2: model2_id})
                if res["code"] == RetCode.SUCCESS:
                    # Clear both (need at least one non-empty field)
                    # Find a third field to keep
                    other_field = "img2txt_id" if field1 != "img2txt_id" and field2 != "img2txt_id" else "llm_id"
                    if builtin_llm:
                        other_model_id = f"{builtin_llm[0]['name']}@Builtin"
                        res = set_default_models(HttpApiAuth, {field1: "", field2: "", other_field: other_model_id})
                        assert res["code"] == RetCode.SUCCESS, res
                        
                        # Verify both were cleared
                        get_res = get_default_models(HttpApiAuth)
                        assert get_res["code"] == RetCode.SUCCESS, get_res
                        assert get_res["data"][field1] == ""
                        assert get_res["data"][field2] == ""
                        return
        
        pytest.skip("Not enough models available for clearing test")

    @pytest.mark.p2
    def test_clear_all_models(self, HttpApiAuth):
        """Test clearing all models (setting all to empty string)"""
        # First set some models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] == RetCode.SUCCESS and "Builtin" in models_res["data"]:
            builtin_llm = models_res["data"]["Builtin"]["llm"]
            if builtin_llm:
                model = builtin_llm[0]
                model_id = f"{model['name']}@Builtin"
                
                # Set one model
                res = set_default_models(HttpApiAuth, {"llm_id": model_id})
                if res["code"] == RetCode.SUCCESS:
                    # Try to clear all - but API requires at least one non-empty field
                    # So we can't actually clear ALL models, but we can clear most
                    # This test documents the API limitation
                    res = set_default_models(HttpApiAuth, {
                        "llm_id": "",
                        "embd_id": "",
                        "img2txt_id": "",
                        "asr_id": "",
                        "rerank_id": "",
                        "tts_id": "",
                    })
                    # Should fail because at least one model ID must be provided
                    assert res["code"] == RetCode.ARGUMENT_ERROR, res
                    assert res["message"] == "At least one model ID must be provided", res
                    return
        
        pytest.skip("No models available for clearing test")

    @pytest.mark.p2
    def test_set_empty_request(self, HttpApiAuth):
        """Test that empty request fails"""
        res = set_default_models(HttpApiAuth, {})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "At least one model ID must be provided", res

    @pytest.mark.p2
    def test_set_none_values(self, HttpApiAuth):
        """Test that None values are treated as empty"""
        res = set_default_models(HttpApiAuth, {"llm_id": None, "embd_id": None})
        # Should fail because all values are None/empty
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "At least one model ID must be provided", res
