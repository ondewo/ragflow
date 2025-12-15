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
from typing import List

import pytest
from pytest import FixtureRequest

from common import remove_model


@pytest.fixture(scope="class")
def cleanup_added_models(request: FixtureRequest, HttpApiAuth):
    """Fixture to clean up models added during tests (for TestAddModel class)"""
    # Track factories that might be added during tests
    factories_to_cleanup: List[str] = ["Builtin", "LocalAI", "Ollama", "Xinference", "LM-Studio", "GPUStack", "FastEmbed"]

    def cleanup():
        # Remove test factories that were added during tests
        for factory in factories_to_cleanup:
            try:
                remove_model(HttpApiAuth, {"llm_factory": factory})
            except Exception:
                # Ignore errors during cleanup (factory might not exist)
                pass

    request.addfinalizer(cleanup)
