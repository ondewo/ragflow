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
"""The Redis client always speaks TLS with a verified server and a client certificate.

``redis.ca_certs``, ``redis.client_cert`` and ``redis.client_key`` are required; a missing or unreadable file stops the start instead of leaving the client unconnected.
The Redis client class is mocked, so no server is needed.
"""

from unittest.mock import Mock

import pytest

import common.settings  # noqa: F401 -- load order of the settings/connector import cycle.
from rag.utils import redis_conn

pytestmark = pytest.mark.p2


def _redis_db_class() -> type:
    """Return the class behind the ``@singleton`` wrapper of ``RedisDB``."""
    for cell in redis_conn.RedisDB.__closure__ or ():
        if isinstance(cell.cell_contents, type):
            return cell.cell_contents
    raise RuntimeError("Could not locate the RedisDB class")


def _open(monkeypatch: pytest.MonkeyPatch, config: dict) -> Mock:
    """Run the real ``__open__`` against a mocked client class; return that mock."""
    client_class = Mock()
    monkeypatch.setattr(redis_conn.redis, "StrictRedis", client_class)
    cls = _redis_db_class()
    db = cls.__new__(cls)
    db.REDIS = None
    db.config = config
    db.__open__()
    return client_class


@pytest.fixture
def tls_files(tmp_path) -> dict[str, str]:
    files = {key: str(tmp_path / f"{key}.pem") for key in ("ca_certs", "client_cert", "client_key")}
    for path in files.values():
        open(path, "w").close()
    return files


def test_client_verifies_the_server_and_presents_its_certificate(monkeypatch, tls_files):
    client_class = _open(monkeypatch, {"host": "valkey:6379", "password": "pw", **tls_files})

    kwargs = client_class.call_args.kwargs
    assert kwargs["host"] == "valkey"
    assert kwargs["port"] == 6379
    assert kwargs["ssl"] is True
    assert kwargs["ssl_ca_certs"] == tls_files["ca_certs"]
    assert kwargs["ssl_certfile"] == tls_files["client_cert"]
    assert kwargs["ssl_keyfile"] == tls_files["client_key"]
    assert kwargs["ssl_cert_reqs"] == "required"
    assert kwargs["ssl_check_hostname"] is True


@pytest.mark.parametrize("key", ["ca_certs", "client_cert", "client_key"])
def test_missing_certificate_setting_is_refused(monkeypatch, tls_files, key):
    config = {"host": "valkey:6379", **tls_files}
    del config[key]

    with pytest.raises(ValueError, match=rf"redis\.{key} is not set"):
        _open(monkeypatch, config)


@pytest.mark.parametrize("key", ["ca_certs", "client_cert", "client_key"])
def test_unreadable_certificate_file_is_refused(monkeypatch, tls_files, tmp_path, key):
    config = {"host": "valkey:6379", **tls_files, key: str(tmp_path / "absent.pem")}

    with pytest.raises(ValueError, match=rf"redis\.{key} in service_conf.yaml names .*absent.pem.*, which cannot be read"):
        _open(monkeypatch, config)
