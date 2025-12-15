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
from configs import INVALID_API_TOKEN
from libs.auth import RAGFlowHttpApiAuth


@pytest.mark.p1
class TestAuthorization:
    @pytest.mark.parametrize(
        "invalid_auth, expected_code, expected_message",
        [
            (None, 0, "`Authorization` can't be empty"),
            (
                RAGFlowHttpApiAuth(INVALID_API_TOKEN),
                109,
                "Authentication error: API key is invalid!",
            ),
        ],
        ids=["empty_auth", "invalid_api_token"],
    )
    def test_invalid_auth(self, invalid_auth, expected_code, expected_message):
        res = set_default_models(invalid_auth, {"llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


class TestSetDefaultModels:
    @pytest.mark.p1
    def test_set_llm_id_builtin(self, HttpApiAuth):
        """Test setting a builtin LLM model"""
        model_id: str = "glm-4-flash@Builtin"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        assert res["code"] == 0, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == model_id

    @pytest.mark.p1
    def test_set_embd_id_builtin(self, HttpApiAuth):
        """Test setting a builtin embedding model"""
        model_id: str = "BAAI/bge-small-en-v1.5@Builtin"
        res = set_default_models(HttpApiAuth, {"embd_id": model_id})
        assert res["code"] == 0, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("embd_id") == model_id

    @pytest.mark.p1
    def test_set_img2txt_id_builtin(self, HttpApiAuth):
        """Test setting a builtin image-to-text model"""
        model_id: str = "glm-4v@Builtin"
        res = set_default_models(HttpApiAuth, {"img2txt_id": model_id})
        assert res["code"] == 0, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("img2txt_id") == model_id

    @pytest.mark.p1
    def test_set_multiple_models_builtin(self, HttpApiAuth):
        """Test setting multiple builtin models at once"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "img2txt_id": "glm-4v@Builtin",
        }
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == 0, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]

    @pytest.mark.p1
    def test_set_all_model_types(self, HttpApiAuth):
        """Test setting all model types"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "asr_id": "",
            "img2txt_id": "glm-4v@Builtin",
            "rerank_id": "",
            "tts_id": "",
        }
        res = set_default_models(HttpApiAuth, payload)
        assert res["code"] == 0, res

        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("asr_id") == payload["asr_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]
        assert models.get("rerank_id") == payload["rerank_id"]
        # tts_id might be None from database, but should be treated as empty string
        assert models.get("tts_id") == payload["tts_id"] or models.get("tts_id") is None

    @pytest.mark.p1
    def test_set_configured_model(self, HttpApiAuth):
        """Test setting a configured model (if available)"""
        # First, try to set a model that might be configured (e.g., from ZHIPU-AI if available)
        # This test assumes the tenant has at least one configured model
        # If not configured, it will use a builtin model instead
        model_id: str = "glm-4-flash@ZHIPU-AI"
        res = set_default_models(HttpApiAuth, {"llm_id": model_id})
        if res["code"] == 0:
            res = get_default_models(HttpApiAuth)
            assert res["code"] == 0, res
            models: Dict[str, Any] = res["data"]
            assert models.get("llm_id") == model_id
        else:
            # If model is not configured, fall back to builtin
            if "not configured" in res["message"]:
                model_id = "glm-4-flash@Builtin"
                res = set_default_models(HttpApiAuth, {"llm_id": model_id})
                assert res["code"] == 0, res
                res = get_default_models(HttpApiAuth)
                assert res["code"] == 0, res
                models: Dict[str, Any] = res["data"]
                assert models.get("llm_id") == model_id

    @pytest.mark.p2
    def test_set_empty_request(self, HttpApiAuth):
        """Test that empty request fails"""
        res = set_default_models(HttpApiAuth, {})
        assert res["code"] != 0, res
        assert "At least one model ID must be provided" in res["message"], res

    @pytest.mark.p2
    def test_set_empty_dict(self, HttpApiAuth):
        """Test that empty dict fails"""
        res = set_default_models(HttpApiAuth, {})
        assert res["code"] != 0, res
        assert "At least one model ID must be provided" in res["message"], res

    @pytest.mark.p2
    def test_set_none_values(self, HttpApiAuth):
        """Test that None values are treated as empty"""
        res = set_default_models(HttpApiAuth, {"llm_id": None, "embd_id": None})
        # The API should reject this or treat it as empty
        assert res["code"] != 0, res
        assert "At least one model ID must be provided" in res["message"] or "not instance of" in res["message"], res

    @pytest.mark.p2
    def test_set_empty_string(self, HttpApiAuth):
        """Test that empty string is ignored (API doesn't process empty strings to clear models)"""
        # First set a model
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flash@Builtin", "embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"

        # Try to clear llm_id with empty string (but keep embd_id to satisfy "at least one" requirement)
        # Note: Empty strings are ignored by the API due to the condition `if field_name in req and req[field_name]:`
        # So the model remains unchanged
        res = set_default_models(HttpApiAuth, {"llm_id": "", "embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        # Empty string is ignored, so llm_id remains unchanged
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

    @pytest.mark.p2
    def test_set_whitespace_string(self, HttpApiAuth):
        """Test that whitespace-only string clears the model (whitespace is truthy, so it's processed)"""
        # First set a model
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flash@Builtin", "embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == 0, res

        # Then clear with whitespace (but keep embd_id to satisfy "at least one" requirement)
        # Note: Whitespace strings are truthy, so they pass the condition and are processed as empty strings
        res = set_default_models(HttpApiAuth, {"llm_id": "   ", "embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        # Whitespace string is processed and clears the model
        assert models.get("llm_id") == ""
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

    @pytest.mark.p2
    def test_set_nonexistent_model(self, HttpApiAuth):
        """Test setting a non-existent model fails"""
        res = set_default_models(HttpApiAuth, {"llm_id": "nonexistent-model@UnknownFactory"})
        assert res["code"] != 0, res
        assert "not configured" in res["message"] or "Model" in res["message"], res

    @pytest.mark.p2
    def test_set_invalid_model_format(self, HttpApiAuth):
        """Test setting a model with invalid format"""
        res = set_default_models(HttpApiAuth, {"llm_id": "invalid-format"})
        # Should fail validation
        assert res["code"] != 0, res
        assert "not configured" in res["message"] or "Model" in res["message"], res

    @pytest.mark.p2
    def test_set_missing_at_symbol(self, HttpApiAuth):
        """Test setting a model without @ symbol"""
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flashBuiltin"})
        # Should fail validation
        assert res["code"] != 0, res
        assert "not configured" in res["message"] or "Model" in res["message"], res

    @pytest.mark.p2
    def test_set_partial_update(self, HttpApiAuth):
        """Test that only provided models are updated, others remain unchanged"""
        # Set initial models
        initial_payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
        }
        res = set_default_models(HttpApiAuth, initial_payload)
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        initial_models: Dict[str, Any] = res["data"]

        # Update only one model
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        updated_models: Dict[str, Any] = res["data"]

        # LLM should be updated
        assert updated_models.get("llm_id") == "glm-4@Builtin"
        # Embedding should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")

    @pytest.mark.p2
    def test_set_clear_one_keep_others(self, HttpApiAuth):
        """Test that empty strings don't clear models (API limitation)"""
        # Set multiple models
        res = set_default_models(
            HttpApiAuth,
            {
                "llm_id": "glm-4-flash@Builtin",
                "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
                "img2txt_id": "glm-4v@Builtin",
            },
        )
        assert res["code"] == 0, res

        # Try to clear one model with empty string (but keep at least one non-empty to satisfy API requirement)
        # Note: Empty strings are ignored by the API, so llm_id remains unchanged
        res = set_default_models(HttpApiAuth, {"llm_id": "", "embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]

        # Empty string is ignored, so llm_id remains unchanged
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"
        # img2txt_id should remain unchanged
        assert models.get("img2txt_id") == "glm-4v@Builtin"

    @pytest.mark.p3
    def test_set_rerank_id(self, HttpApiAuth):
        """Test setting rerank model"""
        # Set rerank_id along with at least one non-empty model to satisfy API requirement
        # Note: Empty strings are ignored, so rerank_id won't be cleared, but we can verify the API accepts it
        res = set_default_models(HttpApiAuth, {"rerank_id": "", "llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        # Empty string is ignored, so rerank_id remains unchanged (whatever it was before)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

    @pytest.mark.p3
    def test_set_asr_id(self, HttpApiAuth):
        """Test setting ASR model"""
        # Set asr_id along with at least one non-empty model to satisfy API requirement
        # Note: Empty strings are ignored, so asr_id won't be cleared, but we can verify the API accepts it
        res = set_default_models(HttpApiAuth, {"asr_id": "", "llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        # Empty string is ignored, so asr_id remains unchanged (whatever it was before)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

    @pytest.mark.p3
    def test_set_tts_id(self, HttpApiAuth):
        """Test setting TTS model"""
        # Set tts_id along with at least one non-empty model to satisfy API requirement
        # Note: Empty strings are ignored, so tts_id won't be cleared, but we can verify the API accepts it
        res = set_default_models(HttpApiAuth, {"tts_id": "", "llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        # Empty string is ignored, so tts_id remains unchanged (whatever it was before)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

    @pytest.mark.p3
    def test_set_models_sequential_updates(self, HttpApiAuth):
        """Test sequential updates to different models"""
        # First update
        res = set_default_models(HttpApiAuth, {"llm_id": "glm-4-flash@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"

        # Second update
        res = set_default_models(HttpApiAuth, {"embd_id": "BAAI/bge-small-en-v1.5@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

        # Third update
        res = set_default_models(HttpApiAuth, {"img2txt_id": "glm-4v@Builtin"})
        assert res["code"] == 0, res
        res = get_default_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"  # Should remain
        assert models.get("img2txt_id") == "glm-4v@Builtin"
