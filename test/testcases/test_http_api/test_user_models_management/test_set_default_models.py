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
        res = set_default_models(invalid_auth, {"llm_id": "test-model@Ollama"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestSetDefaultModelsOllama:
    """Test setting Ollama default models"""

    @pytest.mark.p1
    def test_set_ollama_model_exists(self, HttpApiAuth):
        """Test setting an Ollama model that exists in the system"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Try to find an available model from any factory
        model_id = None
        field = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                if factory_llm:
                    # Use first available model
                    model = factory_llm[0]
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    field_map = {
                        "chat": "llm_id",
                        "embedding": "embd_id",
                        "image2text": "img2txt_id",
                        "speech2text": "asr_id",
                        "rerank": "rerank_id",
                        "tts": "tts_id",
                    }
                    field = field_map.get(model_type, "llm_id")
                    break
        
        # If no models found, try to add Ollama factory
        if not model_id:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS and "Ollama" in models_res["data"]:
                ollama_llm = models_res["data"]["Ollama"]["llm"]
                if ollama_llm:
                    model = ollama_llm[0]
                    model_type = model.get("type")
                    model_id = f"{model['name']}@Ollama"
                    field_map = {
                        "chat": "llm_id",
                        "embedding": "embd_id",
                        "image2text": "img2txt_id",
                        "speech2text": "asr_id",
                        "rerank": "rerank_id",
                        "tts": "tts_id",
                    }
                    field = field_map.get(model_type, "llm_id")
        
        if not model_id:
            pytest.skip("No models available in this RAGFlow instance")
        
        # Set the model as default
        res = set_default_models(HttpApiAuth, {field: model_id})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify it was set
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        assert get_res["data"][field] == model_id

    @pytest.mark.p1
    def test_set_ollama_model_not_exists(self, HttpApiAuth):
        """Test setting an Ollama model that doesn't exist in the system"""
        # First, add Ollama factory
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        
        # Use a model name that definitely doesn't exist
        model_id = "nonexistent-ollama-model@Ollama"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        # Should fail because model doesn't exist
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
    def test_set_ollama_model_by_type(self, HttpApiAuth, model_type, model_field):
        """Test setting models for each model type"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Try to find a model of the requested type from any factory
        model_id = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == model_type:
                        model_id = f"{model['name']}@{factory_name}"
                        break
                if model_id:
                    break
        
        # If no model found, try to add Ollama factory
        if not model_id:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip(f"Could not add Ollama factory and no {model_type} models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            if model.get("type") == model_type:
                                model_id = f"{model['name']}@{factory_name}"
                                break
                        if model_id:
                            break
        
        if not model_id:
            pytest.skip(f"No {model_type} models available in this RAGFlow instance")
        
        payload = {model_field: model_id}
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res
        # Verify it was set
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        assert get_res["data"][model_field] == model_id

    @pytest.mark.p2
    def test_set_multiple_ollama_models_all_valid(self, HttpApiAuth):
        """Test setting multiple models when all are valid"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        payload = {}
        
        # Find models of different types from any available factory
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and "llm_id" not in payload:
                        payload["llm_id"] = model_id
                    elif model_type == "embedding" and "embd_id" not in payload:
                        payload["embd_id"] = model_id
                    elif model_type == "image2text" and "img2txt_id" not in payload:
                        payload["img2txt_id"] = model_id
                    if len(payload) >= 2:
                        break
                if len(payload) >= 2:
                    break
        
        # If not enough models found, try to add Ollama factory
        if len(payload) < 2:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and not enough models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            model_type = model.get("type")
                            model_id = f"{model['name']}@{factory_name}"
                            if model_type == "chat" and "llm_id" not in payload:
                                payload["llm_id"] = model_id
                            elif model_type == "embedding" and "embd_id" not in payload:
                                payload["embd_id"] = model_id
                            elif model_type == "image2text" and "img2txt_id" not in payload:
                                payload["img2txt_id"] = model_id
                            if len(payload) >= 2:
                                break
                        if len(payload) >= 2:
                            break
        
        if len(payload) < 2:
            pytest.skip("Not enough model types available")
        
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify all were set
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        for field, model_id in payload.items():
            assert get_res["data"][field] == model_id

    @pytest.mark.p2
    def test_set_multiple_ollama_models_mix_valid_invalid(self, HttpApiAuth):
        """Test setting multiple Ollama models with mix of valid and invalid"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find any available model from any factory
        valid_model = None
        valid_field = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                if factory_llm:
                    model = factory_llm[0]
                    model_type = model.get("type")
                    valid_model = f"{model['name']}@{factory_name}"
                    field_map = {
                        "chat": "llm_id",
                        "embedding": "embd_id",
                        "image2text": "img2txt_id",
                        "speech2text": "asr_id",
                        "rerank": "rerank_id",
                        "tts": "tts_id",
                    }
                    valid_field = field_map.get(model_type, "llm_id")
                    break
        
        # If no model found, try to add Ollama factory
        if not valid_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        if factory_llm:
                            model = factory_llm[0]
                            model_type = model.get("type")
                            valid_model = f"{model['name']}@{factory_name}"
                            field_map = {
                                "chat": "llm_id",
                                "embedding": "embd_id",
                                "image2text": "img2txt_id",
                                "speech2text": "asr_id",
                                "rerank": "rerank_id",
                                "tts": "tts_id",
                            }
                            valid_field = field_map.get(model_type, "llm_id")
                            break
        
        if not valid_model or not valid_field:
            pytest.skip("No models available")
        
        # Mix valid and invalid
        payload = {
            valid_field: valid_model,
            "llm_id": "nonexistent-model@Ollama",  # Invalid
        }
        
        res = set_default_models(HttpApiAuth, payload)
        # Should fail because one model is invalid
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "not configured" in res["message"], res

    @pytest.mark.p2
    def test_set_multiple_ollama_models_all_invalid(self, HttpApiAuth):
        """Test setting multiple Ollama models when all are invalid"""
        # First, list all available models to check if Ollama factory exists
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Check if Ollama factory exists, if not, add it
        if "Ollama" not in models_res["data"]:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory")
        
        payload = {
            "llm_id": "nonexistent-model1@Ollama",
            "embd_id": "nonexistent-model2@Ollama",
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
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find any available model from any factory
        model_id = None
        field = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                if factory_llm:
                    model = factory_llm[0]
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    field_map = {
                        "chat": "llm_id",
                        "embedding": "embd_id",
                        "image2text": "img2txt_id",
                        "speech2text": "asr_id",
                        "rerank": "rerank_id",
                        "tts": "tts_id",
                    }
                    field = field_map.get(model_type, "llm_id")
                    break
        
        # If no model found, try to add Ollama factory
        if not model_id:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        if factory_llm:
                            model = factory_llm[0]
                            model_type = model.get("type")
                            model_id = f"{model['name']}@{factory_name}"
                            field_map = {
                                "chat": "llm_id",
                                "embedding": "embd_id",
                                "image2text": "img2txt_id",
                                "speech2text": "asr_id",
                                "rerank": "rerank_id",
                                "tts": "tts_id",
                            }
                            field = field_map.get(model_type, "llm_id")
                            break
        
        if not model_id or not field:
            pytest.skip("No configured models available for testing")
        
        # Set it as default
        res = set_default_models(HttpApiAuth, {field: model_id})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify it was set
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        assert get_res["data"][field] == model_id

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
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        payload = {}
        
        # Find models of different types from any available factory
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and "llm_id" not in payload:
                        payload["llm_id"] = model_id
                    elif model_type == "embedding" and "embd_id" not in payload:
                        payload["embd_id"] = model_id
                    if len(payload) >= 2:
                        break
                if len(payload) >= 2:
                    break
        
        # If not enough models found, try to add Ollama factory
        if len(payload) < 2:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and not enough models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            model_type = model.get("type")
                            model_id = f"{model['name']}@{factory_name}"
                            if model_type == "chat" and "llm_id" not in payload:
                                payload["llm_id"] = model_id
                            elif model_type == "embedding" and "embd_id" not in payload:
                                payload["embd_id"] = model_id
                            if len(payload) >= 2:
                                break
                        if len(payload) >= 2:
                            break
        
        if len(payload) < 2:
            pytest.skip("Not enough model types available")
        
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
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find any available model from any factory
        valid_model = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                if factory_llm:
                    valid_model = f"{factory_llm[0]['name']}@{factory_name}"
                    break
        
        # If no model found, try to add Ollama factory
        if not valid_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        if factory_llm:
                            valid_model = f"{factory_llm[0]['name']}@{factory_name}"
                            break
        
        if not valid_model:
            pytest.skip("No models available")
        
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
