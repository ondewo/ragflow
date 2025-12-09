from typing import Any, Dict

import pytest
from conftest import get_default_models, set_default_models
from ragflow_sdk import RAGFlow


class TestGetDefaultModels:
    @pytest.mark.p1
    def test_get_default_models_structure(self, client: RAGFlow) -> None:
        """Test that get_default_models returns all expected fields"""
        models: Dict[str, Any] = get_default_models(client)
        
        # Verify all expected fields are present
        assert "llm_id" in models
        assert "embd_id" in models
        assert "asr_id" in models
        assert "img2txt_id" in models
        assert "rerank_id" in models
        assert "tts_id" in models
        
        # Verify all fields are strings (or None for tts_id)
        assert isinstance(models.get("llm_id"), str) or models.get("llm_id") is None
        assert isinstance(models.get("embd_id"), str) or models.get("embd_id") is None
        assert isinstance(models.get("asr_id"), str) or models.get("asr_id") is None
        assert isinstance(models.get("img2txt_id"), str) or models.get("img2txt_id") is None
        assert isinstance(models.get("rerank_id"), str) or models.get("rerank_id") is None
        assert isinstance(models.get("tts_id"), str) or models.get("tts_id") is None

    @pytest.mark.p1
    def test_get_default_models_empty(self, client: RAGFlow) -> None:
        """Test getting default models returns valid response structure"""
        models: Dict[str, Any] = get_default_models(client)
        
        # Verify the function returns a valid response with all expected fields
        # Values may be empty strings, None (for tts_id), or actual model IDs
        # depending on the initial state
        assert "llm_id" in models
        assert "embd_id" in models
        assert "asr_id" in models
        assert "img2txt_id" in models
        assert "rerank_id" in models
        assert "tts_id" in models
        
        # All values should be strings or None
        for key, value in models.items():
            assert isinstance(value, str) or value is None, f"{key} should be str or None, got {type(value)}"

    @pytest.mark.p1
    def test_get_llm_id_after_set(self, client: RAGFlow) -> None:
        """Test getting LLM model ID after setting it"""
        model_id: str = "glm-4-flash@Builtin"
        set_default_models(client, llm_id=model_id)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == model_id

    @pytest.mark.p1
    def test_get_embd_id_after_set(self, client: RAGFlow) -> None:
        """Test getting embedding model ID after setting it"""
        model_id: str = "BAAI/bge-small-en-v1.5@Builtin"
        set_default_models(client, embd_id=model_id)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("embd_id") == model_id

    @pytest.mark.p1
    def test_get_img2txt_id_after_set(self, client: RAGFlow) -> None:
        """Test getting image-to-text model ID after setting it"""
        model_id: str = "glm-4v@Builtin"
        set_default_models(client, img2txt_id=model_id)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("img2txt_id") == model_id

    @pytest.mark.p1
    def test_get_multiple_models_after_set(self, client: RAGFlow) -> None:
        """Test getting multiple model IDs after setting them"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "img2txt_id": "glm-4v@Builtin",
        }
        set_default_models(client, **payload)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]

    @pytest.mark.p1
    def test_get_all_model_types_after_set(self, client: RAGFlow) -> None:
        """Test getting all model types after setting them"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "asr_id": "",
            "img2txt_id": "glm-4v@Builtin",
            "rerank_id": "",
            "tts_id": "",
        }
        set_default_models(client, **payload)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("asr_id") == payload["asr_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]
        assert models.get("rerank_id") == payload["rerank_id"]
        # tts_id might be None from database, but should be treated as empty string
        assert models.get("tts_id") == payload["tts_id"] or models.get("tts_id") is None

    @pytest.mark.p1
    def test_get_models_reflects_set_operation(self, client: RAGFlow) -> None:
        """Test that get_default_models accurately reflects what was set"""
        # Set initial models
        initial_payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
        }
        set_default_models(client, **initial_payload)
        
        initial_models: Dict[str, Any] = get_default_models(client)
        assert initial_models.get("llm_id") == initial_payload["llm_id"]
        assert initial_models.get("embd_id") == initial_payload["embd_id"]
        
        # Update one model
        set_default_models(client, llm_id="glm-4@Builtin")
        
        updated_models: Dict[str, Any] = get_default_models(client)
        assert updated_models.get("llm_id") == "glm-4@Builtin"
        # Embedding should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")

    @pytest.mark.p1
    def test_get_configured_model(self, client: RAGFlow) -> None:
        """Test getting a configured model (if available)"""
        # First, try to set a model that might be configured (e.g., from ZHIPU-AI if available)
        # If not configured, it will use a builtin model instead
        try:
            model_id: str = "glm-4-flash@ZHIPU-AI"
            set_default_models(client, llm_id=model_id)
            models: Dict[str, Any] = get_default_models(client)
            assert models.get("llm_id") == model_id
        except Exception as e:
            # If model is not configured, fall back to builtin
            if "not configured" in str(e):
                model_id: str = "glm-4-flash@Builtin"
                set_default_models(client, llm_id=model_id)
                models: Dict[str, Any] = get_default_models(client)
                assert models.get("llm_id") == model_id
            else:
                raise

    @pytest.mark.p2
    def test_get_models_after_partial_update(self, client: RAGFlow) -> None:
        """Test getting models after partial update (only some models changed)"""
        # Set initial models
        set_default_models(
            client,
            llm_id="glm-4-flash@Builtin",
            embd_id="BAAI/bge-small-en-v1.5@Builtin",
            img2txt_id="glm-4v@Builtin",
        )
        
        initial_models: Dict[str, Any] = get_default_models(client)
        
        # Update only LLM
        set_default_models(client, llm_id="glm-4@Builtin")
        
        updated_models: Dict[str, Any] = get_default_models(client)
        assert updated_models.get("llm_id") == "glm-4@Builtin"
        # Other models should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")
        assert updated_models.get("img2txt_id") == initial_models.get("img2txt_id")

    @pytest.mark.p2
    def test_get_models_sequential_operations(self, client: RAGFlow) -> None:
        """Test getting models after sequential set operations"""
        # First set
        set_default_models(client, llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        
        # Second set
        set_default_models(client, embd_id="BAAI/bge-small-en-v1.5@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"
        
        # Third set
        set_default_models(client, img2txt_id="glm-4v@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"  # Should remain
        assert models.get("img2txt_id") == "glm-4v@Builtin"

    @pytest.mark.p2
    def test_get_models_consistency(self, client: RAGFlow) -> None:
        """Test that multiple get calls return consistent results"""
        set_default_models(
            client,
            llm_id="glm-4-flash@Builtin",
            embd_id="BAAI/bge-small-en-v1.5@Builtin",
        )
        
        models1: Dict[str, Any] = get_default_models(client)
        models2: Dict[str, Any] = get_default_models(client)
        models3: Dict[str, Any] = get_default_models(client)
        
        # All calls should return the same values
        assert models1 == models2 == models3

    @pytest.mark.p2
    def test_get_models_empty_strings(self, client: RAGFlow) -> None:
        """Test getting models when some are set"""
        # Set some models
        set_default_models(
            client,
            llm_id="glm-4-flash@Builtin",
            embd_id="BAAI/bge-small-en-v1.5@Builtin",
        )
        
        models: Dict[str, Any] = get_default_models(client)
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
    def test_get_models_after_clearing_with_whitespace(self, client: RAGFlow) -> None:
        """Test getting models after clearing one with whitespace string"""
        # Set initial models
        set_default_models(
            client,
            llm_id="glm-4-flash@Builtin",
            embd_id="BAAI/bge-small-en-v1.5@Builtin",
        )
        
        # Clear LLM with whitespace (but keep embd_id to satisfy "at least one" requirement)
        set_default_models(client, llm_id="   ", embd_id="BAAI/bge-small-en-v1.5@Builtin")
        
        models: Dict[str, Any] = get_default_models(client)
        # LLM should be cleared (empty string)
        assert models.get("llm_id") == ""
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

    @pytest.mark.p3
    def test_get_rerank_id(self, client: RAGFlow) -> None:
        """Test getting rerank model ID"""
        # Note: We can't easily set rerank_id without a valid rerank model
        # So we just verify it's in the response
        models: Dict[str, Any] = get_default_models(client)
        assert "rerank_id" in models
        assert isinstance(models.get("rerank_id"), str) or models.get("rerank_id") is None

    @pytest.mark.p3
    def test_get_asr_id(self, client: RAGFlow) -> None:
        """Test getting ASR model ID"""
        # Note: We can't easily set asr_id without a valid ASR model
        # So we just verify it's in the response
        models: Dict[str, Any] = get_default_models(client)
        assert "asr_id" in models
        assert isinstance(models.get("asr_id"), str) or models.get("asr_id") is None

    @pytest.mark.p3
    def test_get_tts_id(self, client: RAGFlow) -> None:
        """Test getting TTS model ID"""
        # Note: We can't easily set tts_id without a valid TTS model
        # So we just verify it's in the response
        models: Dict[str, Any] = get_default_models(client)
        assert "tts_id" in models
        # tts_id might be None from database
        assert isinstance(models.get("tts_id"), str) or models.get("tts_id") is None

    @pytest.mark.p3
    def test_get_models_response_format(self, client: RAGFlow) -> None:
        """Test that get_default_models returns a dictionary with correct format"""
        models: Dict[str, Any] = get_default_models(client)
        
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
    def test_get_models_after_multiple_changes(self, client: RAGFlow) -> None:
        """Test getting models after multiple changes to the same field"""
        # Set LLM to first value
        set_default_models(client, llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        
        # Change LLM to second value
        set_default_models(client, llm_id="glm-4@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4@Builtin"
        
        # Change LLM back to first value
        set_default_models(client, llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

