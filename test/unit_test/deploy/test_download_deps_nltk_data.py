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

"""Tests for the NLTK provisioning guards of the image build.

An NLTK corpus that never downloaded produces an image that boots fine and
fails on the first document parse, because `api/validation.py` re-downloads with
`halt_on_error=False, quiet=True`. Two guards close that gap and are locked
here: `ragflow_deps/download_deps.py::verify_nltk_data` rejects an incomplete
download when the deps image is built, and the `Dockerfile` re-checks the copied
`nltk_data` so a stale `infiniflow/ragflow_deps:latest` cannot slip through
either. The downloader guard is only worth anything while `__main__` still calls
it on the directory the download loop wrote to, so that wiring is pinned here as
well. The Dockerfile assertions also pin the ordering the HTTP/1.1 git pin
depends on; the image build itself is not exercised here.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

# pytest's pythonpath includes "." but be defensive about import location.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ragflow_deps.download_deps import NLTK_RESOURCE_PATHS, verify_nltk_data

DOCKERFILE = _ROOT / "Dockerfile"
DOWNLOAD_DEPS = _ROOT / "ragflow_deps" / "download_deps.py"


def provision(download_dir, resource_path, as_zip):
    """Lay out one NLTK resource the way the downloader leaves it: unzipped as a
    directory, or as the `.zip` NLTK keeps next to it."""
    target = Path(download_dir) / resource_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if as_zip:
        Path(f"{target}.zip").write_bytes(b"PK\x03\x04")
    else:
        target.mkdir()


def dockerfile_lines():
    return DOCKERFILE.read_text(encoding="utf-8").splitlines()


def first_index(lines, predicate, what):
    for index, line in enumerate(lines):
        if predicate(line):
            return index
    raise AssertionError(f"{what} not found in {DOCKERFILE}")


def main_block():
    """The statements of the downloader's `if __name__ == "__main__"` body."""
    for node in ast.parse(DOWNLOAD_DEPS.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            return node.body
    raise AssertionError(f"no __main__ block in {DOWNLOAD_DEPS}")


def walk_all(nodes):
    for node in nodes:
        yield from ast.walk(node)


def calls_to(nodes, name):
    return [node for node in walk_all(nodes) if isinstance(node, ast.Call) and ast.unparse(node.func) == name]


def test_passes_when_every_resource_is_an_unzipped_directory(tmp_path, capsys):
    for resource_path in NLTK_RESOURCE_PATHS.values():
        provision(tmp_path, resource_path, as_zip=False)

    verify_nltk_data(str(tmp_path))

    assert str(tmp_path) in capsys.readouterr().out


def test_passes_when_every_resource_is_a_zip(tmp_path):
    for resource_path in NLTK_RESOURCE_PATHS.values():
        provision(tmp_path, resource_path, as_zip=True)

    verify_nltk_data(str(tmp_path))


def test_passes_when_resources_mix_directories_and_zips(tmp_path):
    # A real nltk_data holds both shapes at once: wordnet ships with unzip=0 and
    # stays a .zip while punkt_tab is extracted, so alternate the two shapes.
    for index, resource_path in enumerate(NLTK_RESOURCE_PATHS.values()):
        provision(tmp_path, resource_path, as_zip=bool(index % 2))

    verify_nltk_data(str(tmp_path))


@pytest.mark.parametrize("absent", sorted(NLTK_RESOURCE_PATHS))
def test_fails_when_one_resource_is_absent(tmp_path, absent):
    for data, resource_path in NLTK_RESOURCE_PATHS.items():
        if data != absent:
            provision(tmp_path, resource_path, as_zip=False)

    with pytest.raises(SystemExit) as excinfo:
        verify_nltk_data(str(tmp_path))

    message = str(excinfo.value)
    assert absent in message
    assert NLTK_RESOURCE_PATHS[absent] in message.replace("\\", "/")
    for data in NLTK_RESOURCE_PATHS:
        if data != absent:
            assert f"{data} (expected" not in message


def test_fails_when_nothing_was_downloaded(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        verify_nltk_data(str(tmp_path))

    message = str(excinfo.value)
    for data in NLTK_RESOURCE_PATHS:
        assert data in message


def test_main_downloads_and_verifies_the_same_resources_and_directory():
    body = main_block()

    download_loops = [node for node in walk_all(body) if isinstance(node, ast.For) and calls_to(node.body, "nltk.download")]
    assert len(download_loops) == 1, "the nltk download loop was moved or duplicated"
    assert ast.unparse(download_loops[0].iter) == "NLTK_RESOURCE_PATHS", "the download loop must iterate NLTK_RESOURCE_PATHS, or the downloaded set drifts from the verified set"

    guard_calls = calls_to(body, "verify_nltk_data")
    assert len(guard_calls) == 1, "verify_nltk_data is not called exactly once from __main__"
    assert guard_calls[0].lineno > download_loops[0].end_lineno, "verify_nltk_data must run after the download loop"

    download_call = calls_to(download_loops[0].body, "nltk.download")[0]
    download_dir = [keyword.value for keyword in download_call.keywords if keyword.arg == "download_dir"]
    assert [ast.unparse(arg) for arg in guard_calls[0].args] == [ast.unparse(node) for node in download_dir], "verify_nltk_data must check the directory the download loop wrote to"


def test_dockerfile_guards_every_resource_download_deps_fetches():
    guard = [line for line in dockerfile_lines() if "for resource in" in line]
    assert len(guard) == 1, "the Dockerfile nltk_data guard was moved or duplicated"

    listed = re.search(r"for resource in (.+?); do", guard[0]).group(1).split()
    assert sorted(listed) == sorted(NLTK_RESOURCE_PATHS.values())


def test_dockerfile_pins_git_to_http_1_1_before_the_first_git_invocation():
    lines = dockerfile_lines()

    git_installed = first_index(lines, lambda line: re.search(r"\bgit\b", line) and "unzip curl wget" in line, "the apt line installing git")
    http_version_pinned = first_index(lines, lambda line: line.strip() == "RUN git config --global http.version HTTP/1.1", "the git HTTP/1.1 pin")
    first_clone = first_index(lines, lambda line: "git clone" in line, "a git clone")
    builder_stage = first_index(lines, lambda line: line.startswith("FROM base AS builder"), "the builder stage")

    assert git_installed < http_version_pinned, "git is not installed yet where the HTTP/1.1 pin runs"
    assert http_version_pinned < first_clone, "the first git clone runs before git is pinned to HTTP/1.1"
    assert http_version_pinned < builder_stage, "the pin must sit in the base stage so derived stages inherit it"
