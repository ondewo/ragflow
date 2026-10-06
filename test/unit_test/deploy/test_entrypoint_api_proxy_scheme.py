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
"""``API_PROXY_SCHEME`` must select the nginx config and the backend coherently.

``docker/entrypoint.sh`` reads ``API_PROXY_SCHEME`` twice: once to pick the
nginx config and once per service to decide whether to start its python or its
go implementation. A value the service guards do not recognise therefore brings
up nginx in front of no backend at all, which answers every request with 502
while the container looks healthy. ``python`` is the fallback for any value
that is neither ``hybrid`` nor ``go``, an unset variable included.

The checks below run the real decision logic: the normalization block and every
``API_PROXY_SCHEME`` guard are extracted verbatim from the script and evaluated
by bash for each candidate value.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"

NGINX_CONF_DIR_LITERAL = "/etc/nginx/conf.d"
# The service guards: a line of the form `if <conditions over API_PROXY_SCHEME>; then`.
_GUARD_RE = re.compile(r"^\s*if\s+(\[\[.*API_PROXY_SCHEME.*\]\])\s*;\s*then\s*$")
# "Applied nginx config: <file>", as the nginx block below echoes it.
_APPLIED_RE = re.compile(r"^Applied nginx config: (\S+)$", re.MULTILINE)


def _script_lines() -> list[str]:
    return ENTRYPOINT.read_text(encoding="utf-8").splitlines()


def _nginx_block_bounds() -> tuple[int, int]:
    """
    Locate the nginx-config selection block, which follows the NGINX_CONF_DIR assignment.

    Returns:
        tuple[int, int]: The line indices of the block's ``if`` and its closing ``fi``.
    """
    lines: list[str] = _script_lines()
    assignments: list[int] = [i for i, line in enumerate(lines) if line.startswith("NGINX_CONF_DIR=")]
    assert len(assignments) == 1, f"expected exactly one NGINX_CONF_DIR assignment, found {len(assignments)}"

    start: int = assignments[0] + 1
    assert lines[start].startswith("if "), f"nginx selection does not start with an if: {lines[start]!r}"

    return start, next(i for i in range(start, len(lines)) if lines[i] == "fi")


def _nginx_block() -> str:
    """
    Extract the nginx-config selection block.

    Returns:
        str: The block as bash source, which picks its directory up from ``$NGINX_CONF_DIR``.
    """
    start, end = _nginx_block_bounds()
    block: str = "\n".join(_script_lines()[start : end + 1])
    assert NGINX_CONF_DIR_LITERAL not in block, "nginx block must reference $NGINX_CONF_DIR, not the literal path"

    return block


def _normalization_block() -> str:
    """
    Extract the statement that resolves ``API_PROXY_SCHEME`` before anything reads it.

    Returns:
        str: The block as bash source.
    """
    lines: list[str] = _script_lines()
    nginx_conf_dir: int = next(i for i, line in enumerate(lines) if line.startswith("NGINX_CONF_DIR="))
    assigns: list[int] = [
        i for i in range(nginx_conf_dir) if lines[i].strip().startswith("API_PROXY_SCHEME=")
    ]
    assert assigns, "entrypoint.sh never resolves API_PROXY_SCHEME before reading it"

    start: int = next(i for i in range(assigns[0], -1, -1) if lines[i].startswith("if "))
    end: int = next(i for i in range(start, len(lines)) if lines[i] == "fi")

    return "\n".join(lines[start : end + 1])


def _service_guards() -> list[str]:
    """
    Extract the per-service guards that decide which backend implementation starts.

    They all sit below the nginx-config selection, which reads the same variable.

    Returns:
        list[str]: Each guard's bash condition, in the order the script evaluates them.
    """
    _, nginx_end = _nginx_block_bounds()

    return [match.group(1) for line in _script_lines()[nginx_end + 1 :] if (match := _GUARD_RE.match(line))]


GUARDS: list[str] = _service_guards()
PYTHON_GUARDS: list[str] = [guard for guard in GUARDS if '"python"' in guard]
GO_GUARDS: list[str] = [guard for guard in GUARDS if '"go"' in guard]


def _run(scheme: str | None, conf_dir: Path) -> tuple[str, list[bool]]:
    """
    Resolve ``scheme`` the way the entrypoint does and report what it would bring up.

    Args:
        scheme (str | None): The value of ``API_PROXY_SCHEME``, or None to leave the variable unset.
        conf_dir (Path): A directory holding the candidate nginx config files.

    Returns:
        tuple[str, list[bool]]: The nginx config applied, and whether each guard in GUARDS passed.
    """
    preamble: str = "" if scheme is None else f"export API_PROXY_SCHEME={scheme!r}\n"
    probes: str = "\n".join(
        f'if {guard}; then echo "guard {i}: yes"; else echo "guard {i}: no"; fi' for i, guard in enumerate(GUARDS)
    )
    script: str = "\n".join(
        [
            preamble,
            f'NGINX_CONF_DIR="{conf_dir}"',
            _normalization_block(),
            _nginx_block(),
            probes,
            "",
        ]
    )

    completed: subprocess.CompletedProcess[str] = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )

    applied: list[str] = _APPLIED_RE.findall(completed.stdout)
    assert len(applied) == 1, f"expected one applied nginx config, got {applied}: {completed.stdout}"
    passed: list[bool] = [f"guard {i}: yes" in completed.stdout for i in range(len(GUARDS))]

    return applied[0], passed


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    for name in ("ragflow.conf.python", "ragflow.conf.hybrid", "ragflow.conf.golang"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")

    return tmp_path


def test_every_guard_is_classified() -> None:
    """A guard naming neither stack would be silently skipped by the checks below."""
    assert len(GUARDS) >= 8, f"expected the per-service guards to be found, got {GUARDS}"
    assert sorted(PYTHON_GUARDS + GO_GUARDS) == sorted(GUARDS)
    assert not set(PYTHON_GUARDS) & set(GO_GUARDS)


@pytest.mark.parametrize("scheme", [None, "", "python", "golang"])
def test_anything_but_hybrid_or_go_runs_the_python_stack(scheme: str | None, conf_dir: Path) -> None:
    applied, passed = _run(scheme=scheme, conf_dir=conf_dir)

    assert applied == "ragflow.conf.python"
    assert [GUARDS[i] for i, ok in enumerate(passed) if ok] == PYTHON_GUARDS


def test_hybrid_runs_both_stacks(conf_dir: Path) -> None:
    applied, passed = _run(scheme="hybrid", conf_dir=conf_dir)

    assert applied == "ragflow.conf.hybrid"
    assert all(passed), f"hybrid must start every service: {[GUARDS[i] for i, ok in enumerate(passed) if not ok]}"


def test_go_runs_the_go_stack(conf_dir: Path) -> None:
    applied, passed = _run(scheme="go", conf_dir=conf_dir)

    assert applied == "ragflow.conf.golang"
    assert [GUARDS[i] for i, ok in enumerate(passed) if ok] == GO_GUARDS


def test_script_parses() -> None:
    subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)
