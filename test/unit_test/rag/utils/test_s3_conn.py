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

import importlib
from unittest.mock import ANY, Mock

import pytest
from botocore.exceptions import ClientError

# rag.utils.s3_conn and common.settings import each other (s3_conn needs
# settings.S3; settings re-imports RAGFlowS3). The cycle resolves only when
# common.settings is imported first, the same order the app uses. Importing it
# here before s3_conn lets both modules load fully so other tests that depend
# on the real settings (e.g. db_models reads settings.DATABASE_TYPE) keep
# working in the same session.
import common.settings  # noqa: F401 (imported for load-order side effect: breaks the s3_conn <-> settings cycle)
from rag.utils import s3_conn

# Readable stand-ins for the certificate files: no test opens a connection, so they are never loaded.
TLS_FILES = {"ca_certs": __file__, "client_cert": __file__, "client_key": __file__}


def _new_storage(monkeypatch, config):
    # Re-import before reloading: an earlier test may have dropped "rag.utils.s3_conn" from sys.modules (see test_s3_conn_hardening.py).
    module = importlib.reload(importlib.import_module("rag.utils.s3_conn"))
    client = Mock()
    monkeypatch.setattr(module.settings, "S3", {**TLS_FILES, **config})
    monkeypatch.setattr(module.boto3, "client", Mock(return_value=client))
    return module.RAGFlowS3(), client, module.boto3.client


def test_s3_accepts_region_config_key(monkeypatch):
    storage, _, client_factory = _new_storage(monkeypatch, {"region": "us-east-1", "bucket": "ragflow"})

    assert storage.region_name == "us-east-1"
    # config carries the request timeouts and retry policy, covered in
    # test_s3_conn_hardening.py.
    client_factory.assert_called_once_with("s3", region_name="us-east-1", config=ANY, verify=TLS_FILES["ca_certs"])


def test_s3_health_uses_head_bucket_without_writing(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})

    assert storage.health() is True
    client.head_bucket.assert_called_once_with(Bucket="ragflow")
    client.create_bucket.assert_not_called()
    client.upload_fileobj.assert_not_called()


def test_s3_health_uses_list_buckets_without_default_bucket(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {})

    assert storage.health() is True
    client.list_buckets.assert_called_once_with()


def test_s3_health_accepts_a_default_bucket_the_first_upload_creates(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})
    client.head_bucket.side_effect = s3_conn.ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    assert storage.health() is True
    client.create_bucket.assert_not_called()


def test_s3_health_returns_false_when_the_bucket_is_forbidden(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})
    client.head_bucket.side_effect = s3_conn.ClientError({"Error": {"Code": "403"}}, "HeadBucket")

    assert storage.health() is False


def test_s3_health_returns_false_on_client_error(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})
    client.head_bucket.side_effect = ConnectionError("unavailable")

    assert storage.health() is False


def _client_error(code):
    return ClientError({"Error": {"Code": code}}, "GetObject")


def test_s3_put_succeeds_on_first_attempt(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})

    assert storage.put("kb", "file.txt", b"data") is client.upload_fileobj.return_value
    client.upload_fileobj.assert_called_once()


def test_s3_get_succeeds_on_first_attempt(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})
    client.get_object.return_value = {"Body": Mock(read=Mock(return_value=b"data"))}

    assert storage.get("kb", "file.txt") == b"data"
    client.get_object.assert_called_once()


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_s3_get_missing_object_codes_return_none_without_retry(monkeypatch, code):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "ragflow"})
    client.get_object.side_effect = _client_error(code)
    reconnect = Mock(return_value=True)
    monkeypatch.setattr(storage, "__open__", reconnect)

    assert storage.get("kb", "missing.txt") is None
    client.get_object.assert_called_once_with(Bucket="ragflow", Key="missing.txt")
    reconnect.assert_not_called()
