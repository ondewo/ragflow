#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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

"""Unit tests for RAGFlowMinio.bucket_exists under a configured default bucket.

``use_default_bucket`` forwards the caller's logical bucket as the
``_orig_bucket`` keyword, so the method has to accept it: without that, account
deletion (which calls bucket_exists per knowledge base) fails with a TypeError
before the body runs.
"""

import importlib
from unittest.mock import Mock

import pytest

from rag.utils import minio_conn

pytestmark = pytest.mark.p2


def _new_storage(monkeypatch, config):
    # Re-import before reloading: an earlier test in the same session may have dropped
    # "rag.utils.minio_conn" from sys.modules, and importlib.reload() raises ImportError on a module
    # that is no longer there. import_module puts it back first, so the fixture does not
    # depend on collection order.
    module = importlib.reload(importlib.import_module("rag.utils.minio_conn"))
    conn = Mock()
    monkeypatch.setattr(module.settings, "MINIO", {"host": "minio.invalid", "user": "u", "password": "p", **config})
    monkeypatch.setattr(module, "Minio", Mock(return_value=conn))
    return module.RAGFlowMinio(), conn


def test_bucket_exists_checks_the_physical_bucket_in_single_bucket_mode(monkeypatch):
    storage, conn = _new_storage(monkeypatch, {"bucket": "shared", "prefix_path": "documents"})
    conn.bucket_exists.return_value = True

    assert storage.bucket_exists("kb-1") is True
    conn.bucket_exists.assert_called_once_with("shared")


def test_bucket_exists_reports_a_missing_physical_bucket_in_single_bucket_mode(monkeypatch):
    storage, conn = _new_storage(monkeypatch, {"bucket": "shared"})
    conn.bucket_exists.return_value = False

    assert storage.bucket_exists("kb-1") is False


def test_bucket_exists_checks_the_named_bucket_in_multi_bucket_mode(monkeypatch):
    storage, conn = _new_storage(monkeypatch, {})
    conn.bucket_exists.return_value = True

    assert storage.bucket_exists("kb-1") is True
    conn.bucket_exists.assert_called_once_with("kb-1")
