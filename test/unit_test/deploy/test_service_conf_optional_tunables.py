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
"""The optional s3 and os tunables documented in both service_conf files.

`conf/service_conf.yaml` is the reference config and
`docker/service_conf.yaml.template` is what `docker/entrypoint.sh` renders at
container start-up. Both document the same optional keys, so the two must not
drift apart, the documented defaults must match the ones the code applies when a
key is absent, and the keys must stay commented out so an existing
`service_conf.yaml` keeps working.
"""

from pathlib import Path
import re

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[3]
SERVICE_CONF = ROOT / "conf" / "service_conf.yaml"
SERVICE_CONF_TEMPLATE = ROOT / "docker" / "service_conf.yaml.template"
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# The defaults the readers of settings.S3 / settings.OS apply when the key is absent.
S3_TUNABLE_DEFAULTS = {"connect_timeout": 10, "read_timeout": 60, "retries_max_attempts": 5}
OS_TUNABLE_DEFAULTS = {"pool_size": 32}


def _load_yaml(text):
    return YAML(typ="safe", pure=True).load(text)


def _render(template_text):
    """Substitute ${VAR:-default} placeholders the way docker/entrypoint.sh does with no env set."""
    return _PLACEHOLDER_PATTERN.sub(lambda match: match.group(2) or "", template_text)


def _config_texts():
    """The two config files as YAML text, the template in its rendered form."""
    return {
        "conf/service_conf.yaml": SERVICE_CONF.read_text(encoding="utf-8"),
        "docker/service_conf.yaml.template": _render(SERVICE_CONF_TEMPLATE.read_text(encoding="utf-8")),
    }


def _uncommented_s3_block(text):
    """The commented-out `# s3:` block as live YAML, so its keys can be parsed."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == "# s3:")

    block = ["s3:"]
    for line in lines[start + 1 :]:
        if not line.startswith("#   "):
            break
        block.append(line[2:])

    return "\n".join(block) + "\n"


def _os_block(text, uncomment_optionals):
    """The live `os:` block, optionally with its commented-out `key: value` lines enabled."""
    block = []
    for line in text.splitlines():
        if not block:
            if line.strip() == "os:":
                block.append(line)
            continue
        if line and not line.startswith(" "):
            break
        stripped = line.strip()
        if uncomment_optionals and stripped.startswith("# ") and ":" in stripped:
            block.append("  " + stripped[2:])
        else:
            block.append(line)

    return "\n".join(block) + "\n"


@pytest.mark.parametrize("name", sorted(_config_texts()))
def test_config_parses_as_yaml(name):
    config = _load_yaml(_config_texts()[name])

    assert config["os"]["hosts"]
    assert config["ragflow"]["http_port"] == 9380


@pytest.mark.parametrize("name", sorted(_config_texts()))
def test_optional_tunables_stay_commented_out(name):
    config = _load_yaml(_config_texts()[name])

    assert "s3" not in config
    assert set(OS_TUNABLE_DEFAULTS).isdisjoint(config["os"])


@pytest.mark.parametrize("name", sorted(_config_texts()))
def test_documented_s3_tunables_carry_the_code_defaults(name):
    s3 = _load_yaml(_uncommented_s3_block(_config_texts()[name]))["s3"]

    assert {key: s3[key] for key in S3_TUNABLE_DEFAULTS} == S3_TUNABLE_DEFAULTS
    # prefix_path is what scopes a logical bucket inside a shared physical bucket.
    assert s3["prefix_path"]


@pytest.mark.parametrize("name", sorted(_config_texts()))
def test_documented_os_tunables_carry_the_code_defaults(name):
    os_block = _load_yaml(_os_block(_config_texts()[name], uncomment_optionals=True))["os"]

    assert {key: os_block[key] for key in OS_TUNABLE_DEFAULTS} == OS_TUNABLE_DEFAULTS


def test_both_configs_document_the_same_optional_tunables():
    texts = _config_texts()
    conf, template = texts["conf/service_conf.yaml"], texts["docker/service_conf.yaml.template"]

    conf_s3, template_s3 = _load_yaml(_uncommented_s3_block(conf))["s3"], _load_yaml(_uncommented_s3_block(template))["s3"]
    documented = set(S3_TUNABLE_DEFAULTS) | {"prefix_path"}
    assert {key: conf_s3[key] for key in documented} == {key: template_s3[key] for key in documented}

    conf_os = _load_yaml(_os_block(conf, uncomment_optionals=True))["os"]
    template_os = _load_yaml(_os_block(template, uncomment_optionals=True))["os"]
    assert {key: conf_os[key] for key in OS_TUNABLE_DEFAULTS} == {key: template_os[key] for key in OS_TUNABLE_DEFAULTS}
