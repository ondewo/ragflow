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
from common import get_default_models, set_default_models
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
        res = set_default_models(invalid_auth, {"llm_id": "glm-4-flash@Builtin"})
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
        model_id: str = "glm-4-flash@Builtin"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == model_id

    @pytest.mark.p1
    def test_get_embd_id_after_set(self, HttpApiAuth):
        """Test getting embedding model ID after setting it"""
        model_id: str = "BAAI/bge-small-en-v1.5@Builtin"
        res = set_default_models(HttpApiAuth, {"embd_id": model_id})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("embd_id") == model_id

    @pytest.mark.p1
    def test_get_img2txt_id_after_set(self, HttpApiAuth):
        """Test getting image-to-text model ID after setting it"""
        model_id: str = "glm-4v@Builtin"
        res = set_default_models(HttpApiAuth, {"img2txt_id": model_id})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("img2txt_id") == model_id

    @pytest.mark.p1
    def test_get_multiple_models_after_set(self, HttpApiAuth):
        """Test getting multiple model IDs after setting them"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "img2txt_id": "glm-4v@Builtin",
        }
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]

    @pytest.mark.p1
    def test_get_all_model_types_after_set(self, HttpApiAuth):
        """Test getting all model types after setting them"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "asr_id": "",
            "img2txt_id": "glm-4v@Builtin",
            "rerank_id": "",
            "tts_id": "",
        }
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == RetCode.SUCCESS, res

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
        # Set initial models
        initial_payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
        }
        res = set_default_models(HttpApiAuth, initial_payload)
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        initial_models: Dict[str, Any] = res["data"]
        assert initial_models.get("llm_id") == initial_payload["llm_id"]
        assert initial_models.get("embd_id") == initial_payload["embd_id"]

        # Update one model
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        updated_models: Dict[str, Any] = res["data"]
        assert updated_models.get("llm_id") == "glm-4@Builtin"
        # Embedding should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")

    @pytest.mark.p1
    def test_get_configured_model(self, HttpApiAuth):
        """Test getting a configured model (if available)"""
        # First, try to set a model that might be configured (e.g., from ZHIPU-AI if available)
        # If not configured, it will use a builtin model instead
        model_id: str = "glm-4-flash@ZHIPU-AI"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        
        # Check response code - either success or argument error
        assert res["code"] in [RetCode.SUCCESS, RetCode.ARGUMENT_ERROR], res
        
        if res["code"] == RetCode.SUCCESS:
            # Model is configured, verify it can be retrieved
            get_res = get_default_models(HttpApiAuth)
            assert get_res["code"] == RetCode.SUCCESS, get_res
            models: Dict[str, Any] = get_res["data"]
            assert models.get("llm_id") == model_id
        else:
            # Model is not configured, verify error message and fall back to builtin
            assert res["message"] == f"Model '{model_id}' (type: chat) is not configured. Please add the model first using POST /api/v1/models", res
            model_id = "glm-4-flash@Builtin"
            set_res = set_default_models(HttpApiAuth, {"llm_id": model_id})
            assert set_res["code"] == RetCode.SUCCESS, set_res
            get_res = get_default_models(HttpApiAuth)
            assert get_res["code"] == RetCode.SUCCESS, get_res
            models: Dict[str, Any] = get_res["data"]
            assert models.get("llm_id") == model_id

    @pytest.mark.p2
    def test_get_models_after_partial_update(self, HttpApiAuth):
        """Test getting models after partial update (only some models changed)"""
        # Set initial models
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": "glm-4-flash@Builtin",
                "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
                "img2txt_id": "glm-4v@Builtin",
            },
        )
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        initial_models: Dict[str, Any] = res["data"]

        # Update only LLM
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        updated_models: Dict[str, Any] = res["data"]
        assert updated_models.get("llm_id") == "glm-4@Builtin"
        # Other models should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")
        assert updated_models.get("img2txt_id") == initial_models.get("img2txt_id")

    @pytest.mark.p2
    def test_get_models_sequential_operations(self, HttpApiAuth):
        """Test getting models after sequential set operations"""
        # First set
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"

        # Second set
        res = set_default_models(HttpApiAuth, {"embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

        # Third set
        res = set_default_models(HttpApiAuth, {"img2txt_id": "glm-4v@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"  # Should remain
        assert models.get("img2txt_id") == "glm-4v@Builtin"

    @pytest.mark.p2
    def test_get_models_consistency(self, HttpApiAuth):
        """Test that multiple get calls return consistent results"""
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": "glm-4-flash@Builtin",
                "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
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
        # Set some models
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": "glm-4-flash@Builtin",
                "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            },
        )
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        # Verify the models we set are correct
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"
        # Other models may be empty strings, None (for tts_id), or have values from previous tests
        # Just verify they are valid types
        assert isinstance(models.get("asr_id"), str) or models.get("asr_id") is None
        assert isinstance(models.get("img2txt_id"), str) or models.get("img2txt_id") is None
        assert isinstance(models.get("rerank_id"), str) or models.get("rerank_id") is None
        assert isinstance(models.get("tts_id"), str) or models.get("tts_id") is None

    @pytest.mark.p2
    def test_get_models_after_clearing_with_whitespace(self, HttpApiAuth):
        """Test getting models after clearing one with whitespace string"""
        # Set initial models
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": "glm-4-flash@Builtin",
                "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            },
        )
        assert res["code"] == RetCode.SUCCESS, res

        # Clear LLM with whitespace (but keep embd_id to satisfy "at least one" requirement)
        res = set_default_models(HttpApiAuth, {"llm_id": "   ", "embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        # LLM should be cleared (empty string)
        assert models.get("llm_id") == ""
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

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
        # Set LLM to first value
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"

        # Change LLM to second value
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4@Builtin"

        # Change LLM back to first value
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == RetCode.SUCCESS, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"
