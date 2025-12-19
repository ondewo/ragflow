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
    def test_get_default_models_invalid_auth(self, invalid_auth, expected_code, expected_message):
        res = get_default_models(invalid_auth)
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res

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
    def test_set_default_models_invalid_auth(self, invalid_auth, expected_code, expected_message):
        res = set_default_models(invalid_auth, {"llm_id": "test-model@Ollama"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


class TestGetDefaultModels:
    @pytest.mark.p1
    def test_get_default_models_structure(self, HttpApiAuth):
        """Test that get_default_models returns exactly the expected fields and nothing else"""
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]

        # Define exactly the expected keys
        expected_keys = {"llm_id", "embd_id", "asr_id", "img2txt_id", "rerank_id", "tts_id"}
        
        # Verify the response contains exactly these keys and nothing else
        actual_keys = set(models.keys())
        assert actual_keys == expected_keys, f"Expected keys {expected_keys}, but got {actual_keys}"

        # Verify all fields are strings (or None for tts_id)
        assert isinstance(models.get("llm_id"), str) or models.get("llm_id") is None
        assert isinstance(models.get("embd_id"), str) or models.get("embd_id") is None
        assert isinstance(models.get("asr_id"), str) or models.get("asr_id") is None
        assert isinstance(models.get("img2txt_id"), str) or models.get("img2txt_id") is None
        assert isinstance(models.get("rerank_id"), str) or models.get("rerank_id") is None
        assert isinstance(models.get("tts_id"), str) or models.get("tts_id") is None

    @pytest.mark.p1
    def test_get_llm_id_after_set(self, HttpApiAuth):
        """Test getting LLM model ID after setting it"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Try to find a chat model from any available factory
        chat_model = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "chat":
                        chat_model = f"{model['name']}@{factory_name}"
                        break
                if chat_model:
                    break
        
        # If no chat model found, try to add Ollama factory
        if not chat_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no chat models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            if model.get("type") == "chat":
                                chat_model = f"{model['name']}@{factory_name}"
                                break
                        if chat_model:
                            break
        
        if not chat_model:
            pytest.skip("No chat models available in this RAGFlow instance")
        
        # Set it as default
        res = set_default_models(HttpApiAuth, {"llm_id": chat_model})
        assert res["code"] == RetCode.SUCCESS, res

        # Get and verify default models
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == chat_model

    @pytest.mark.p1
    def test_get_embd_id_after_set(self, HttpApiAuth):
        """Test getting embedding model ID after setting it"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Try to find an embedding model from any available factory
        embd_model = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "embedding":
                        embd_model = f"{model['name']}@{factory_name}"
                        break
                if embd_model:
                    break
        
        # If no embedding model found, try to add Ollama factory
        if not embd_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no embedding models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            if model.get("type") == "embedding":
                                embd_model = f"{model['name']}@{factory_name}"
                                break
                        if embd_model:
                            break
        
        if not embd_model:
            pytest.skip("No embedding models available in this RAGFlow instance")
        
        # Set it as default
        res = set_default_models(HttpApiAuth, {"embd_id": embd_model})
        assert res["code"] == RetCode.SUCCESS, res

        # Get and verify default models
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("embd_id") == embd_model

    @pytest.mark.p1
    def test_get_img2txt_id_after_set(self, HttpApiAuth):
        """Test getting image-to-text model ID after setting it"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Try to find an image2text model from any available factory
        img2txt_model = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "image2text":
                        img2txt_model = f"{model['name']}@{factory_name}"
                        break
                if img2txt_model:
                    break
        
        # If no image2text model found, try to add Ollama factory
        if not img2txt_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no image2text models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            if model.get("type") == "image2text":
                                img2txt_model = f"{model['name']}@{factory_name}"
                                break
                        if img2txt_model:
                            break
        
        if not img2txt_model:
            pytest.skip("No image2text models available in this RAGFlow instance")
        
        # Set it as default
        res = set_default_models(HttpApiAuth, {"img2txt_id": img2txt_model})
        assert res["code"] == RetCode.SUCCESS, res

        # Get and verify default models
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("img2txt_id") == img2txt_model

    @pytest.mark.p1
    def test_get_multiple_models_after_set(self, HttpApiAuth):
        """Test getting multiple model IDs after setting them"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find models of different types from any available factory
        payload: Dict[str, str] = {}
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
                    if len(payload) >= 3:
                        break
                if len(payload) >= 3:
                    break
        
        # If not enough models found, try to add Ollama factory
        if len(payload) < 3:
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
                            if len(payload) >= 3:
                                break
                        if len(payload) >= 3:
                            break
        
        if len(payload) < 3:
            pytest.skip("Not enough model types available")
        
        # Set them as default
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res

        # Get and verify default models
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]

    @pytest.mark.p1
    def test_get_all_model_types_after_set(self, HttpApiAuth):
        """Test getting all model types after setting them"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find models of different types from any available factory
        payload: Dict[str, str] = {
            "asr_id": "",
            "rerank_id": "",
            "tts_id": "",
        }
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
                    if "llm_id" in payload and "embd_id" in payload and "img2txt_id" in payload:
                        break
                if "llm_id" in payload and "embd_id" in payload and "img2txt_id" in payload:
                    break
        
        # If not enough models found, try to add Ollama factory
        if "llm_id" not in payload or "embd_id" not in payload or "img2txt_id" not in payload:
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
                            if "llm_id" in payload and "embd_id" in payload and "img2txt_id" in payload:
                                break
                        if "llm_id" in payload and "embd_id" in payload and "img2txt_id" in payload:
                            break
        
        if "llm_id" not in payload or "embd_id" not in payload or "img2txt_id" not in payload:
            pytest.skip("Not enough model types available")
        
        # Set them as default
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res

        # Get and verify default models
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("asr_id") == payload["asr_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]
        assert models.get("rerank_id") == payload["rerank_id"]
        # tts_id might be None from database, but should be treated as empty string
        assert models.get("tts_id") == payload["tts_id"] or models.get("tts_id") is None

    @pytest.mark.p1
    def test_get_models_reflects_set_operation(self, HttpApiAuth):
        """Test that get_default_models accurately reflects what was set"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find initial models from any available factory
        initial_payload: Dict[str, str] = {}
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and "llm_id" not in initial_payload:
                        initial_payload["llm_id"] = model_id
                    elif model_type == "embedding" and "embd_id" not in initial_payload:
                        initial_payload["embd_id"] = model_id
                    if "llm_id" in initial_payload and "embd_id" in initial_payload:
                        break
                if "llm_id" in initial_payload and "embd_id" in initial_payload:
                    break
        
        # If not enough models found, try to add Ollama factory
        if "llm_id" not in initial_payload or "embd_id" not in initial_payload:
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
                            if model_type == "chat" and "llm_id" not in initial_payload:
                                initial_payload["llm_id"] = model_id
                            elif model_type == "embedding" and "embd_id" not in initial_payload:
                                initial_payload["embd_id"] = model_id
                            if "llm_id" in initial_payload and "embd_id" in initial_payload:
                                break
                        if "llm_id" in initial_payload and "embd_id" in initial_payload:
                            break
        
        if "llm_id" not in initial_payload or "embd_id" not in initial_payload:
            pytest.skip("Not enough model types available")
        
        # Set initial models
        res = set_default_models(HttpApiAuth, initial_payload)
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        initial_models: Dict[str, Any] = res["data"]
        assert initial_models.get("llm_id") == initial_payload["llm_id"]
        assert initial_models.get("embd_id") == initial_payload["embd_id"]

        # Update one model - find another chat model from any factory
        updated_llm_id = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "chat":
                        candidate_id = f"{model['name']}@{factory_name}"
                        if candidate_id != initial_payload["llm_id"]:
                            updated_llm_id = candidate_id
                            break
                if updated_llm_id:
                    break
        
        if not updated_llm_id:
            pytest.skip("No alternative chat model available")
        
        res = set_default_models(HttpApiAuth, {"llm_id": updated_llm_id})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        updated_models: Dict[str, Any] = res["data"]
        assert updated_models.get("llm_id") == updated_llm_id
        # Embedding should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")

    @pytest.mark.p1
    def test_get_configured_model(self, HttpApiAuth):
        """Test getting a configured model (if available)"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Try to find a chat model from any available factory
        chat_model = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "chat":
                        chat_model = f"{model['name']}@{factory_name}"
                        break
                if chat_model:
                    break
        
        # If no chat model found, try to add Ollama factory
        if not chat_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no chat models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            if model.get("type") == "chat":
                                chat_model = f"{model['name']}@{factory_name}"
                                break
                        if chat_model:
                            break
        
        if not chat_model:
            pytest.skip("No chat models available")
        
        # Set it as default
        res = set_default_models(HttpApiAuth, {"llm_id": chat_model})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify it can be retrieved
        get_res = get_default_models(HttpApiAuth)
        assert get_res["code"] == RetCode.SUCCESS, get_res
        models: Dict[str, Any] = get_res["data"]
        assert models.get("llm_id") == chat_model

    @pytest.mark.p2
    def test_get_models_after_partial_update(self, HttpApiAuth):
        """Test getting models after partial update (only some models changed)"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find initial models from any available factory
        initial_payload: Dict[str, str] = {}
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and "llm_id" not in initial_payload:
                        initial_payload["llm_id"] = model_id
                    elif model_type == "embedding" and "embd_id" not in initial_payload:
                        initial_payload["embd_id"] = model_id
                    elif model_type == "image2text" and "img2txt_id" not in initial_payload:
                        initial_payload["img2txt_id"] = model_id
                    if "llm_id" in initial_payload and "embd_id" in initial_payload and "img2txt_id" in initial_payload:
                        break
                if "llm_id" in initial_payload and "embd_id" in initial_payload and "img2txt_id" in initial_payload:
                    break
        
        # If not enough models found, try to add Ollama factory
        if "llm_id" not in initial_payload or "embd_id" not in initial_payload or "img2txt_id" not in initial_payload:
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
                            if model_type == "chat" and "llm_id" not in initial_payload:
                                initial_payload["llm_id"] = model_id
                            elif model_type == "embedding" and "embd_id" not in initial_payload:
                                initial_payload["embd_id"] = model_id
                            elif model_type == "image2text" and "img2txt_id" not in initial_payload:
                                initial_payload["img2txt_id"] = model_id
                            if "llm_id" in initial_payload and "embd_id" in initial_payload and "img2txt_id" in initial_payload:
                                break
                        if "llm_id" in initial_payload and "embd_id" in initial_payload and "img2txt_id" in initial_payload:
                            break
        
        if "llm_id" not in initial_payload or "embd_id" not in initial_payload or "img2txt_id" not in initial_payload:
            pytest.skip("Not enough model types available")
        
        # Set initial models
        res = set_default_models(HttpApiAuth, initial_payload)
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        initial_models: Dict[str, Any] = res["data"]

        # Update only LLM - find another chat model from any factory
        updated_llm_id = None
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "chat":
                        candidate_id = f"{model['name']}@{factory_name}"
                        if candidate_id != initial_payload["llm_id"]:
                            updated_llm_id = candidate_id
                            break
                if updated_llm_id:
                    break
        
        if not updated_llm_id:
            pytest.skip("No alternative chat model available")
        
        res = set_default_models(HttpApiAuth, {"llm_id": updated_llm_id})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        updated_models: Dict[str, Any] = res["data"]
        assert updated_models.get("llm_id") == updated_llm_id
        # Other models should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")
        assert updated_models.get("img2txt_id") == initial_models.get("img2txt_id")

    @pytest.mark.p2
    def test_get_models_sequential_operations(self, HttpApiAuth):
        """Test getting models after sequential set operations"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find models of different types from any available factory
        chat_model = None
        embd_model = None
        img2txt_model = None
        
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and not chat_model:
                        chat_model = model_id
                    elif model_type == "embedding" and not embd_model:
                        embd_model = model_id
                    elif model_type == "image2text" and not img2txt_model:
                        img2txt_model = model_id
        
        # If no models found, try to add Ollama factory
        if not chat_model:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and no chat models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            model_type = model.get("type")
                            model_id = f"{model['name']}@{factory_name}"
                            if model_type == "chat" and not chat_model:
                                chat_model = model_id
                            elif model_type == "embedding" and not embd_model:
                                embd_model = model_id
                            elif model_type == "image2text" and not img2txt_model:
                                img2txt_model = model_id
        
        if not chat_model:
            pytest.skip("No chat models available")
        
        # First set
        res = set_default_models(HttpApiAuth, {"llm_id": chat_model})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == chat_model

        # Second set (if embedding model available)
        if embd_model:
            res = set_default_models(HttpApiAuth, {"embd_id": embd_model})
            assert res["code"] == RetCode.SUCCESS, res
            res = get_default_models(HttpApiAuth)
            assert res["code"] == RetCode.SUCCESS, res
            models: Dict[str, Any] = res["data"]
            assert models.get("llm_id") == chat_model  # Should remain
            assert models.get("embd_id") == embd_model

        # Third set (if image2text model available)
        if img2txt_model:
            res = set_default_models(HttpApiAuth, {"img2txt_id": img2txt_model})
            assert res["code"] == RetCode.SUCCESS, res
            res = get_default_models(HttpApiAuth)
            assert res["code"] == RetCode.SUCCESS, res
            models: Dict[str, Any] = res["data"]
            assert models.get("llm_id") == chat_model  # Should remain
            if embd_model:
                assert models.get("embd_id") == embd_model  # Should remain
            assert models.get("img2txt_id") == img2txt_model

    @pytest.mark.p2
    def test_get_models_consistency(self, HttpApiAuth):
        """Test that multiple get calls return consistent results"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find models from any available factory
        chat_model = None
        embd_model = None
        
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and not chat_model:
                        chat_model = model_id
                    elif model_type == "embedding" and not embd_model:
                        embd_model = model_id
        
        # If not enough models found, try to add Ollama factory
        if not chat_model or not embd_model:
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
                            if model_type == "chat" and not chat_model:
                                chat_model = model_id
                            elif model_type == "embedding" and not embd_model:
                                embd_model = model_id
        
        if not chat_model or not embd_model:
            pytest.skip("Not enough model types available")
        
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": chat_model,
                "embd_id": embd_model,
            },
        )
        assert res["code"] == RetCode.SUCCESS, res

        res1 = get_default_models(HttpApiAuth)
        assert res1["code"] == RetCode.SUCCESS, res1
        models1: Dict[str, Any] = res1["data"]

        res2 = get_default_models(HttpApiAuth)
        assert res2["code"] == RetCode.SUCCESS, res2
        models2: Dict[str, Any] = res2["data"]

        res3 = get_default_models(HttpApiAuth)
        assert res3["code"] == RetCode.SUCCESS, res3
        models3: Dict[str, Any] = res3["data"]

        # All calls should return the same values
        assert models1 == models2 == models3

    @pytest.mark.p2
    def test_get_models_empty_strings(self, HttpApiAuth):
        """Test getting models when some are set"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find models from any available factory
        chat_model = None
        embd_model = None
        
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and not chat_model:
                        chat_model = model_id
                    elif model_type == "embedding" and not embd_model:
                        embd_model = model_id
        
        # If not enough models found, try to add Ollama factory
        if not chat_model or not embd_model:
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
                            if model_type == "chat" and not chat_model:
                                chat_model = model_id
                            elif model_type == "embedding" and not embd_model:
                                embd_model = model_id
        
        if not chat_model or not embd_model:
            pytest.skip("Not enough model types available")
        
        # Set some models
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": chat_model,
                "embd_id": embd_model,
            },
        )
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        # Verify the models we set are correct
        assert models.get("llm_id") == chat_model
        assert models.get("embd_id") == embd_model
        # Other models may be empty strings, None (for tts_id), or have values from previous tests
        # Just verify they are valid types
        assert isinstance(models.get("asr_id"), str) or models.get("asr_id") is None
        assert isinstance(models.get("img2txt_id"), str) or models.get("img2txt_id") is None
        assert isinstance(models.get("rerank_id"), str) or models.get("rerank_id") is None
        assert isinstance(models.get("tts_id"), str) or models.get("tts_id") is None

    @pytest.mark.p2
    def test_get_models_after_clearing_with_whitespace(self, HttpApiAuth):
        """Test getting models after clearing one with whitespace string"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find models from any available factory
        chat_model = None
        embd_model = None
        
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    model_type = model.get("type")
                    model_id = f"{model['name']}@{factory_name}"
                    if model_type == "chat" and not chat_model:
                        chat_model = model_id
                    elif model_type == "embedding" and not embd_model:
                        embd_model = model_id
        
        # If not enough models found, try to add Ollama factory
        if not chat_model or not embd_model:
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
                            if model_type == "chat" and not chat_model:
                                chat_model = model_id
                            elif model_type == "embedding" and not embd_model:
                                embd_model = model_id
        
        if not chat_model or not embd_model:
            pytest.skip("Not enough model types available")
        
        # Set initial models
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": chat_model,
                "embd_id": embd_model,
            },
        )
        assert res["code"] == RetCode.SUCCESS, res

        # Clear LLM with whitespace (but keep embd_id to satisfy "at least one" requirement)
        res = set_default_models(HttpApiAuth, {"llm_id": "   ", "embd_id": embd_model})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        # LLM should be cleared (empty string)
        assert models.get("llm_id") == ""
        assert models.get("embd_id") == embd_model

    @pytest.mark.p3
    def test_get_rerank_id(self, HttpApiAuth):
        """Test getting rerank model ID"""
        # Note: We can't easily set rerank_id without a valid rerank model
        # So we just verify it's in the response
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert "rerank_id" in models
        assert isinstance(models.get("rerank_id"), str) or models.get("rerank_id") is None

    @pytest.mark.p3
    def test_get_asr_id(self, HttpApiAuth):
        """Test getting ASR model ID"""
        # Note: We can't easily set asr_id without a valid ASR model
        # So we just verify it's in the response
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert "asr_id" in models
        assert isinstance(models.get("asr_id"), str) or models.get("asr_id") is None

    @pytest.mark.p3
    def test_get_tts_id(self, HttpApiAuth):
        """Test getting TTS model ID"""
        # Note: We can't easily set tts_id without a valid TTS model
        # So we just verify it's in the response
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert "tts_id" in models
        # tts_id might be None from database
        assert isinstance(models.get("tts_id"), str) or models.get("tts_id") is None

    @pytest.mark.p3
    def test_get_models_response_format(self, HttpApiAuth):
        """Test that get_default_models returns a dictionary with correct format"""
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]

        # Should be a dictionary
        assert isinstance(models, dict)

        # Should have exactly 6 keys
        assert len(models) == 6

        # All keys should be strings
        for key in models.keys():
            assert isinstance(key, str)

        # All values should be strings or None
        for value in models.values():
            assert isinstance(value, str) or value is None

    @pytest.mark.p3
    def test_get_models_after_multiple_changes(self, HttpApiAuth):
        """Test getting models after multiple changes to the same field"""
        # First, list all available models
        models_res = list_user_models(HttpApiAuth)
        if models_res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not list user models")
        
        # Find multiple chat models from any available factory
        chat_models = []
        for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
            if factory_name in models_res["data"]:
                factory_llm = models_res["data"][factory_name]["llm"]
                for model in factory_llm:
                    if model.get("type") == "chat":
                        chat_models.append(f"{model['name']}@{factory_name}")
        
        # If not enough models found, try to add Ollama factory
        if len(chat_models) < 2:
            res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            if res["code"] != RetCode.SUCCESS:
                pytest.skip("Could not add Ollama factory and not enough chat models available")
            
            # List models again after adding
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == RetCode.SUCCESS:
                for factory_name in ["Ollama", "OpenAI", "ZHIPU-AI"]:
                    if factory_name in models_res["data"]:
                        factory_llm = models_res["data"][factory_name]["llm"]
                        for model in factory_llm:
                            if model.get("type") == "chat":
                                model_id = f"{model['name']}@{factory_name}"
                                if model_id not in chat_models:
                                    chat_models.append(model_id)
        
        if len(chat_models) < 2:
            pytest.skip("Not enough chat models available for multiple changes test")
        
        first_model = chat_models[0]
        second_model = chat_models[1]
        
        # Set LLM to first value
        res = set_default_models(HttpApiAuth, {"llm_id": first_model})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == first_model

        # Change LLM to second value
        res = set_default_models(HttpApiAuth, {"llm_id": second_model})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == second_model

        # Change LLM back to first value
        res = set_default_models(HttpApiAuth, {"llm_id": first_model})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == first_model
