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

"""Unit tests for the bounded, non-swallowing RAGFlowS3 I/O path.

Covers the request timeouts and retries handed to botocore, the one-off
creation of a configured default bucket, and the failure contract of put
(raises), get and get_presigned_url (return None, one attempt).
"""

import importlib
import logging
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

# Import common.settings first: s3_conn <-> settings import cycle resolves
# only in the same order the app uses (see test_s3_conn.py).
import common.settings  # noqa: F401
from rag.utils import s3_conn

pytestmark = pytest.mark.p2


def _new_storage(monkeypatch, config, client=None):
    """Build a fresh S3 singleton over a mock boto3 client.

    Reloading the module drops the per-class singleton cache so every test
    gets its own instance, and no test touches the network.
    """
    # Re-import before reloading: an earlier test in the same session may have dropped
    # "rag.utils.s3_conn" from sys.modules, and importlib.reload() raises ImportError on a module
    # that is no longer there. import_module puts it back first, so the fixture does not
    # depend on collection order.
    module = importlib.reload(importlib.import_module("rag.utils.s3_conn"))
    if client is None:
        client = Mock()
        client.meta.region_name = "eu-central-1"
    monkeypatch.setattr(module.settings, "S3", config)
    monkeypatch.setattr(module.boto3, "client", Mock(return_value=client))
    return module.RAGFlowS3(), client, module


def _mock_client(region="eu-central-1"):
    client = Mock()
    client.meta.region_name = region
    return client


def _client_error(code):
    return ClientError({"Error": {"Code": code}}, "HeadBucket")


# ---------------------------------------------------------------- _int_setting


FALLBACK = 7


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, FALLBACK),
        ({"timeout": None}, FALLBACK),
        ({"timeout": ""}, FALLBACK),
        ({"timeout": 0}, FALLBACK),
        ({"timeout": -3}, FALLBACK),
        ({"timeout": "nonsense"}, FALLBACK),
        ({"timeout": [1]}, FALLBACK),
        ({"timeout": 15}, 15),
        ({"timeout": "15"}, 15),
    ],
)
def test_int_setting_reads_an_optional_positive_integer(config, expected):
    assert s3_conn._int_setting(config, "timeout", FALLBACK) == expected


def test_int_setting_warns_about_an_unusable_value(caplog):
    with caplog.at_level(logging.WARNING):
        assert s3_conn._int_setting({"read_timeout": "soon"}, "read_timeout", 60) == 60

    assert "s3.read_timeout" in caplog.text


# -------------------------------------------------------- timeouts and retries


def test_open_applies_the_documented_timeout_and_retry_defaults(monkeypatch):
    """A deployment that configures none of the keys still gets bounded requests."""
    _, _, module = _new_storage(monkeypatch, {})

    config = module.boto3.client.call_args.kwargs["config"]
    assert config.connect_timeout == 10
    assert config.read_timeout == 60
    assert config.retries == {"max_attempts": 5, "mode": "standard"}
    # botocore's own default is 10, which measurably discards connections once the
    # task executors and API workers upload concurrently.
    assert config.max_pool_connections == 32


def test_open_applies_configured_timeouts_and_retries(monkeypatch):
    storage, _, module = _new_storage(
        monkeypatch,
        {"connect_timeout": 3, "read_timeout": "7", "retries_max_attempts": 2},
    )

    assert (storage.connect_timeout, storage.read_timeout, storage.retries_max_attempts) == (3, 7, 2)
    config = module.boto3.client.call_args.kwargs["config"]
    assert config.connect_timeout == 3
    assert config.read_timeout == 7
    assert config.retries == {"max_attempts": 2, "mode": "standard"}


def test_open_applies_a_configured_pool_size(monkeypatch):
    """The connection pool is sized from the config, like the other bounds."""
    storage, _, module = _new_storage(monkeypatch, {"pool_size": "48"})

    assert storage.pool_size == 48
    assert module.boto3.client.call_args.kwargs["config"].max_pool_connections == 48


def test_open_ignores_an_unusable_pool_size(monkeypatch):
    """A malformed value falls back to the default instead of failing construction."""
    storage, _, module = _new_storage(monkeypatch, {"pool_size": "not-a-number"})

    assert storage.pool_size == 32
    assert module.boto3.client.call_args.kwargs["config"].max_pool_connections == 32


def test_open_keeps_signature_and_addressing_settings_alongside_the_timeouts(monkeypatch):
    _, _, module = _new_storage(monkeypatch, {"signature_version": "s3v4", "addressing_style": "path"})

    config = module.boto3.client.call_args.kwargs["config"]
    assert config.signature_version == "s3v4"
    assert config.s3 == {"addressing_style": "path"}
    assert config.connect_timeout == 10


def test_open_ignores_unusable_timeout_configuration(monkeypatch):
    _, _, module = _new_storage(monkeypatch, {"connect_timeout": "soon", "retries_max_attempts": 0})

    config = module.boto3.client.call_args.kwargs["config"]
    assert config.connect_timeout == 10
    assert config.retries == {"max_attempts": 5, "mode": "standard"}


# ------------------------------------------------------- default bucket, once


def test_construction_issues_no_request(monkeypatch):
    """Building the connector must not block on the object store: it happens in
    init_settings, in every process, before anything is uploaded."""
    client = _mock_client()
    client.head_bucket.side_effect = _client_error("404")

    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared", "prefix_path": "documents"}, client)

    assert storage.bucket == "shared"
    client.head_bucket.assert_not_called()
    client.create_bucket.assert_not_called()


def test_default_bucket_is_created_once_and_not_per_upload(monkeypatch):
    client = _mock_client()
    client.head_bucket.side_effect = _client_error("404")
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared", "prefix_path": "documents"}, client)

    storage.put("kb-1", "file.txt", b"payload")
    storage.put("kb-2", "file.txt", b"payload")

    # The physical bucket is probed and created on the first upload only.
    client.head_bucket.assert_called_once_with(Bucket="shared")
    client.create_bucket.assert_called_once_with(Bucket="shared", CreateBucketConfiguration={"LocationConstraint": "eu-central-1"})
    assert client.upload_fileobj.call_count == 2
    keys = [call.args[2] for call in client.upload_fileobj.call_args_list]
    buckets = {call.args[1] for call in client.upload_fileobj.call_args_list}
    assert keys == ["documents/kb-1/file.txt", "documents/kb-2/file.txt"]
    assert buckets == {"shared"}


def test_existing_default_bucket_is_not_recreated(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared"})

    storage.put("kb-1", "file.txt", b"payload")
    storage.put("kb-2", "file.txt", b"payload")

    client.head_bucket.assert_called_once_with(Bucket="shared")
    client.create_bucket.assert_not_called()
    assert client.upload_fileobj.call_count == 2


def test_an_unreachable_default_bucket_is_retried_on_the_next_upload(monkeypatch):
    """A store that is down when the first upload arrives must not leave the
    bucket uncreated for the lifetime of the process."""
    client = _mock_client()
    client.head_bucket.side_effect = _client_error("500")
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared"}, client)

    with pytest.raises(ClientError):
        storage.put("kb-1", "file.txt", b"payload")
    client.upload_fileobj.assert_not_called()

    client.head_bucket.side_effect = None
    storage.put("kb-1", "file.txt", b"payload")

    assert client.head_bucket.call_count == 2
    assert client.upload_fileobj.call_count == 1


def test_multi_bucket_mode_creates_each_bucket_on_demand(monkeypatch):
    """Without a default bucket, upstream's create-on-first-use is unchanged."""
    client = _mock_client(region="us-east-1")
    client.head_bucket.side_effect = _client_error("404")
    storage, client, _ = _new_storage(monkeypatch, {}, client)

    storage.put("kb-1", "file.txt", b"payload")

    client.head_bucket.assert_called_once_with(Bucket="kb-1")
    # us-east-1 takes no location constraint.
    client.create_bucket.assert_called_once_with(Bucket="kb-1")
    assert client.upload_fileobj.call_args.args[1] == "kb-1"


# --------------------------------------------------------- failure contracts


def test_put_raises_instead_of_reporting_a_successful_upload(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared"})
    client.upload_fileobj.side_effect = RuntimeError("object store is wedged")
    monkeypatch.setattr(storage, "__open__", Mock())

    with pytest.raises(RuntimeError, match="object store is wedged"):
        storage.put("kb-1", "file.txt", b"payload")

    # One attempt, no reconnect: retrying is botocore's job now.
    assert client.upload_fileobj.call_count == 1
    storage.__open__.assert_not_called()


def test_put_raises_when_the_bucket_cannot_be_created(monkeypatch):
    client = _mock_client()
    client.head_bucket.side_effect = _client_error("404")
    storage, client, _ = _new_storage(monkeypatch, {}, client)
    client.create_bucket.side_effect = RuntimeError("no permission")

    with pytest.raises(RuntimeError, match="no permission"):
        storage.put("kb-1", "file.txt", b"payload")

    client.upload_fileobj.assert_not_called()


def test_get_returns_the_object_body(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared", "prefix_path": "documents"})
    body = Mock()
    body.read.return_value = b"payload"
    client.get_object.return_value = {"Body": body}

    assert storage.get("kb-1", "file.txt") == b"payload"
    client.get_object.assert_called_once_with(Bucket="shared", Key="documents/kb-1/file.txt")


def test_get_returns_none_on_failure_without_reconnecting(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared"})
    client.get_object.side_effect = _client_error("NoSuchKey")
    monkeypatch.setattr(storage, "__open__", Mock())

    assert storage.get("kb-1", "file.txt") is None
    assert client.get_object.call_count == 1
    storage.__open__.assert_not_called()


def test_get_presigned_url_returns_the_url(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared", "prefix_path": "documents"})
    client.generate_presigned_url.return_value = "https://s3.invalid/signed"

    assert storage.get_presigned_url("kb-1", "file.txt", 60) == "https://s3.invalid/signed"
    client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "shared", "Key": "documents/kb-1/file.txt"},
        ExpiresIn=60,
    )


def test_get_presigned_url_returns_none_after_a_single_attempt(monkeypatch):
    storage, client, _ = _new_storage(monkeypatch, {"bucket": "shared"})
    client.generate_presigned_url.side_effect = RuntimeError("cannot sign")
    monkeypatch.setattr(storage, "__open__", Mock())

    assert storage.get_presigned_url("kb-1", "file.txt", 60) is None
    assert client.generate_presigned_url.call_count == 1
    storage.__open__.assert_not_called()
