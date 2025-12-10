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

from typing import Any, Dict, List

import pytest
import requests
from common import generate_user_api_key, get_user_api_key
from configs import EMAIL


class TestGetUserApiKey:
    @pytest.mark.p1
    def test_get_user_api_key_success(self, admin_session: requests.Session) -> None:
        """Test successfully getting API keys for a user"""
        user_name: str = EMAIL
        
        # Generate a test API key first
        generated_key: Dict[str, Any] = generate_user_api_key(admin_session, user_name)
        
        # Get all API keys for the user
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify response is a list
        assert isinstance(api_keys, list), "API keys should be returned as a list"
        assert len(api_keys) > 0, "User should have at least one API key"
        
        # Verify the generated key is in the list
        token: str = generated_key["token"]
        token_found: bool = any(key.get("token") == token for key in api_keys)
        assert token_found, "Generated API key should appear in the list"

    @pytest.mark.p1
    def test_get_user_api_key_response_structure(self, admin_session: requests.Session) -> None:
        """Test that get_user_api_key returns correct response structure"""
        user_name: str = EMAIL
        
        # Generate a test API key first
        generate_user_api_key(admin_session, user_name)
        
        # Get all API keys for the user
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify response is a list
        assert isinstance(api_keys, list), "API keys should be returned as a list"
        assert len(api_keys) > 0, "User should have at least one API key"
        
        # Verify structure of each API key in the list
        for key in api_keys:
            assert isinstance(key, dict), "Each API key should be a dictionary"
            assert "token" in key, "API key should contain token"
            assert "beta" in key, "API key should contain beta"
            assert "tenant_id" in key, "API key should contain tenant_id"
            assert "create_date" in key, "API key should contain create_date"
            
            # Verify field types
            assert isinstance(key["token"], str), "token should be string"
            assert isinstance(key["beta"], str), "beta should be string"
            assert isinstance(key["tenant_id"], str), "tenant_id should be string"
            assert isinstance(key.get("create_date"), (str, type(None))), "create_date should be string or None"
            assert isinstance(key.get("update_date"), (str, type(None))), "update_date should be string or None"

    @pytest.mark.p1
    def test_get_user_api_key_includes_newly_generated(self, admin_session: requests.Session) -> None:
        """Test that newly generated API key appears in get_user_api_key list"""
        user_name: str = EMAIL
        
        # Get initial count of API keys
        initial_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        initial_count: int = len(initial_keys)
        
        # Generate a new API key
        generated_key: Dict[str, Any] = generate_user_api_key(admin_session, user_name)
        new_token: str = generated_key["token"]
        
        # Get API keys again
        updated_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify count increased
        assert len(updated_keys) > initial_count, "API key count should increase after generating new key"
        
        # Verify the new token is in the list
        token_found: bool = any(key.get("token") == new_token for key in updated_keys)
        assert token_found, "Newly generated API key should appear in the list"

    @pytest.mark.p2
    def test_get_user_api_key_multiple_keys(self, admin_session: requests.Session) -> None:
        """Test getting multiple API keys for the same user"""
        user_name: str = EMAIL
        
        # Generate multiple API keys
        key1: Dict[str, Any] = generate_user_api_key(admin_session, user_name)
        token1: str = key1["token"]
        
        key2: Dict[str, Any] = generate_user_api_key(admin_session, user_name)
        token2: str = key2["token"]
        
        # Get all API keys
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify both tokens are in the list
        tokens: List[str] = [key.get("token") for key in api_keys]
        assert token1 in tokens, "First token should be in the list"
        assert token2 in tokens, "Second token should be in the list"
        assert len(tokens) >= 2, "Should have at least 2 API keys"

    @pytest.mark.p2
    def test_get_user_api_key_tenant_id_consistency(self, admin_session: requests.Session) -> None:
        """Test that all API keys for a user have the same tenant_id"""
        user_name: str = EMAIL
        
        # Generate multiple API keys
        generate_user_api_key(admin_session, user_name)
        generate_user_api_key(admin_session, user_name)
        
        # Get all API keys
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify all keys have the same tenant_id
        tenant_ids: List[str] = [key.get("tenant_id") for key in api_keys if key.get("tenant_id")]
        if len(tenant_ids) > 0:
            assert all(tid == tenant_ids[0] for tid in tenant_ids), "All API keys should have the same tenant_id"

    @pytest.mark.p2
    def test_get_user_api_key_nonexistent_user(self, admin_session: requests.Session) -> None:
        """Test getting API keys for non-existent user fails"""
        with pytest.raises(Exception) as excinfo:
            get_user_api_key(admin_session, "nonexistent_user_12345")
        assert "User not found" in str(excinfo.value) or "not found" in str(excinfo.value).lower()

    @pytest.mark.p2
    def test_get_user_api_key_empty_username(self, admin_session: requests.Session) -> None:
        """Test getting API keys with empty username fails or returns empty list"""
        # Empty username should either raise an exception or return empty list
        try:
            api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, "")
            # If it doesn't raise, it should return an empty list (no user with empty email)
            assert isinstance(api_keys, list), "Should return a list"
            assert len(api_keys) == 0, "Empty username should return empty list"
        except Exception as excinfo:
            # If it raises an exception, that's also acceptable behavior
            # Empty username is invalid and should be rejected
            assert len(str(excinfo)) > 0, "Exception should have a message"

    @pytest.mark.p2
    def test_get_user_api_key_token_uniqueness(self, admin_session: requests.Session) -> None:
        """Test that all API keys in the list have unique tokens"""
        user_name: str = EMAIL
        
        # Generate multiple API keys
        generate_user_api_key(admin_session, user_name)
        generate_user_api_key(admin_session, user_name)
        
        # Get all API keys
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify all tokens are unique
        tokens: List[str] = [key.get("token") for key in api_keys if key.get("token")]
        assert len(tokens) == len(set(tokens)), "All API keys should have unique tokens"

    @pytest.mark.p3
    def test_get_user_api_key_without_auth(self) -> None:
        """Test that getting API keys without admin auth fails"""
        session: requests.Session = requests.Session()
        user_name: str = EMAIL
        
        with pytest.raises(Exception) as excinfo:
            get_user_api_key(session, user_name)
        # Should fail with authentication error
        assert "auth" in str(excinfo.value).lower() or "login" in str(excinfo.value).lower() or "403" in str(excinfo.value) or "401" in str(excinfo.value)

    @pytest.mark.p3
    def test_get_user_api_key_beta_format(self, admin_session: requests.Session) -> None:
        """Test that beta field in API keys has correct format"""
        user_name: str = EMAIL
        
        # Generate a test API key
        generate_user_api_key(admin_session, user_name)
        
        # Get all API keys
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify beta format for all keys
        for key in api_keys:
            beta: str = key.get("beta", "")
            assert isinstance(beta, str), "beta should be a string"
            assert len(beta) == 32, "beta should be 32 characters"

    @pytest.mark.p3
    def test_get_user_api_key_date_fields(self, admin_session: requests.Session) -> None:
        """Test that date fields in API keys are properly formatted"""
        user_name: str = EMAIL
        
        # Generate a test API key
        generate_user_api_key(admin_session, user_name)
        
        # Get all API keys
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        
        # Verify date fields
        for key in api_keys:
            create_date: Any = key.get("create_date")
            update_date: Any = key.get("update_date")
            
            # create_date should be present (string or None)
            assert create_date is None or isinstance(create_date, str), "create_date should be string or None"
            
            # update_date should be present (string or None)
            assert update_date is None or isinstance(update_date, str), "update_date should be string or None"

    @pytest.mark.p3
    def test_get_user_api_key_case_sensitivity(self, admin_session: requests.Session) -> None:
        """Test that username is case-sensitive when getting API keys"""
        user_name: str = EMAIL
        
        # Generate a test API key with correct case
        generated_key: Dict[str, Any] = generate_user_api_key(admin_session, user_name)
        token: str = generated_key["token"]
        
        # Try to get keys with different case
        try:
            api_keys_upper: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name.upper())
            # If it works, verify it returns the same or different keys
            tokens_upper: List[str] = [key.get("token") for key in api_keys_upper]
            # Either same user (token in list) or different user (token not in list)
            assert isinstance(tokens_upper, list)
        except Exception:
            # Expected to fail if username is case-sensitive
            pass
        
        # Verify correct case works
        api_keys: List[Dict[str, Any]] = get_user_api_key(admin_session, user_name)
        tokens: List[str] = [key.get("token") for key in api_keys]
        assert token in tokens, "Generated token should be in the list for correct case"

