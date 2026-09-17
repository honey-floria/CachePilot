"""加载并校验固定版本的模型与依赖基线。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_FLOATING_NAMES = frozenset({"latest", "main", "master", "dev", "nightly"})


class BaselineError(ValueError):
    """当已提交的基线可能漂移或内部无效时抛出。"""


@dataclass(frozen=True)
class ModelBaseline:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    license_spdx_id: str
    model_max_context_tokens: int
    service_context_limit: int
    model_type: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    trust_remote_code: bool


@dataclass(frozen=True)
class DependencyProfile:
    name: str
    platform: str
    python: str
    torch: Optional[str]
    transformers: Optional[str]
    vllm: Optional[str]
    status: str
    packages: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class DependencyBaseline:
    python_version: str
    profiles: Tuple[DependencyProfile, ...]


def load_model_baseline(path: Path) -> ModelBaseline:
    payload = _load_json(path)
    _require_schema_version(payload, path)

    model_revision = _required_string(payload, "model_revision", path)
    tokenizer_revision = _required_string(payload, "tokenizer_revision", path)
    _validate_revision(model_revision, "model_revision")
    _validate_revision(tokenizer_revision, "tokenizer_revision")

    license_payload = _required_mapping(payload, "license", path)
    architecture = _required_mapping(payload, "architecture", path)

    model_max_context_tokens = _required_positive_int(
        payload, "model_max_context_tokens", path
    )
    service_context_limit = _required_positive_int(
        payload, "initial_service_context_limit", path
    )
    if service_context_limit > model_max_context_tokens:
        raise BaselineError(
            "initial_service_context_limit cannot exceed model_max_context_tokens"
        )

    hidden_size = _required_positive_int(architecture, "hidden_size", path)
    num_attention_heads = _required_positive_int(
        architecture, "num_attention_heads", path
    )
    head_dim = _required_positive_int(architecture, "head_dim", path)
    if hidden_size % num_attention_heads != 0:
        raise BaselineError("hidden_size must be divisible by num_attention_heads")
    if hidden_size // num_attention_heads != head_dim:
        raise BaselineError("head_dim must equal hidden_size / num_attention_heads")

    trust_remote_code = payload.get("trust_remote_code")
    if type(trust_remote_code) is not bool:
        raise BaselineError("trust_remote_code must be a boolean")

    _reject_floating_values(payload, path)

    return ModelBaseline(
        model_id=_required_string(payload, "model_id", path),
        model_revision=model_revision,
        tokenizer_id=_required_string(payload, "tokenizer_id", path),
        tokenizer_revision=tokenizer_revision,
        license_spdx_id=_required_string(license_payload, "spdx_id", path),
        model_max_context_tokens=model_max_context_tokens,
        service_context_limit=service_context_limit,
        model_type=_required_string(architecture, "model_type", path),
        num_hidden_layers=_required_positive_int(
            architecture, "num_hidden_layers", path
        ),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=_required_positive_int(
            architecture, "num_key_value_heads", path
        ),
        hidden_size=hidden_size,
        head_dim=head_dim,
        trust_remote_code=trust_remote_code,
    )


def load_dependency_baseline(path: Path) -> DependencyBaseline:
    payload = _load_json(path)
    _require_schema_version(payload, path)
    _reject_floating_values(payload, path)

    python_version = _required_string(payload, "python_version", path)
    _validate_version(python_version, "python_version")

    raw_profiles = _required_mapping(payload, "profiles", path)
    profiles = []
    for name, raw_profile in sorted(raw_profiles.items()):
        if not isinstance(raw_profile, Mapping):
            raise BaselineError("profile {0} must be an object".format(name))

        profile_python = _required_string(raw_profile, "python", path)
        _validate_version(profile_python, "profiles.{0}.python".format(name))
        if profile_python != python_version:
            raise BaselineError(
                "profile {0} must use the baseline Python version".format(name)
            )

        package_versions = []
        for package_name in (
            "torch",
            "torchaudio",
            "torchvision",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface_hub",
            "vllm",
        ):
            version = raw_profile.get(package_name)
            if version is None:
                continue
            if type(version) is not str:
                raise BaselineError(
                    "profiles.{0}.{1} must be a string or null".format(
                        name, package_name
                    )
                )
            _validate_version(version, "profiles.{0}.{1}".format(name, package_name))
            package_versions.append((package_name, version))

        profiles.append(
            DependencyProfile(
                name=name,
                platform=_required_string(raw_profile, "platform", path),
                python=profile_python,
                torch=_optional_string(raw_profile, "torch"),
                transformers=_optional_string(raw_profile, "transformers"),
                vllm=_optional_string(raw_profile, "vllm"),
                status=_required_string(raw_profile, "status", path),
                packages=tuple(package_versions),
            )
        )

    if not profiles:
        raise BaselineError("at least one dependency profile is required")

    return DependencyBaseline(
        python_version=python_version,
        profiles=tuple(profiles),
    )


def validate_repository_baselines(repository_root: Path) -> None:
    model = load_model_baseline(repository_root / "config" / "model.json")
    dependencies = load_dependency_baseline(
        repository_root / "config" / "dependencies.json"
    )

    if model.model_revision != model.tokenizer_revision:
        raise BaselineError(
            "the initial model and tokenizer must use the same repository snapshot"
        )

    python_version_file = (repository_root / ".python-version").read_text(
        encoding="utf-8"
    ).strip()
    if python_version_file != dependencies.python_version:
        raise BaselineError(".python-version does not match dependencies.json")

    constraints = _parse_constraints(
        repository_root / "requirements" / "constraints-gpu.txt"
    )
    vllm_profile = next(
        profile for profile in dependencies.profiles if profile.name == "vllm_executor"
    )
    for package_name, version in vllm_profile.packages:
        constraint_name = package_name.replace("_", "-")
        if constraint_name not in constraints:
            raise BaselineError(
                "constraints-gpu.txt is missing {0}".format(constraint_name)
            )
        if constraints[constraint_name] != version:
            raise BaselineError(
                "constraint for {0} does not match dependencies.json".format(
                    constraint_name
                )
            )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError("cannot load {0}: {1}".format(path, exc)) from exc
    if not isinstance(payload, Mapping):
        raise BaselineError("{0} must contain a JSON object".format(path))
    return payload


def _require_schema_version(payload: Mapping[str, Any], path: Path) -> None:
    if payload.get("schema_version") != 1:
        raise BaselineError("{0} must use schema_version 1".format(path))


def _required_mapping(
    payload: Mapping[str, Any], key: str, path: Path
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BaselineError("{0}:{1} must be an object".format(path, key))
    return value


def _required_string(payload: Mapping[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if type(value) is not str or not value:
        raise BaselineError("{0}:{1} must be a non-empty string".format(path, key))
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not str or not value:
        raise BaselineError("{0} must be a non-empty string or null".format(key))
    return value


def _required_positive_int(
    payload: Mapping[str, Any], key: str, path: Path
) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 1:
        raise BaselineError("{0}:{1} must be a positive integer".format(path, key))
    return value


def _validate_revision(value: str, field: str) -> None:
    if not _REVISION_PATTERN.fullmatch(value):
        raise BaselineError("{0} must be a 40-character Git commit SHA".format(field))


def _validate_version(value: str, field: str) -> None:
    if not _VERSION_PATTERN.fullmatch(value):
        raise BaselineError("{0} must be an exact x.y.z version".format(field))


def _reject_floating_values(value: Any, path: Path) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_floating_values(child, path)
    elif isinstance(value, list):
        for child in value:
            _reject_floating_values(child, path)
    elif isinstance(value, str) and value.lower() in _FLOATING_NAMES:
        raise BaselineError(
            "{0} contains a floating revision or version: {1}".format(path, value)
        )


def _parse_constraints(path: Path) -> Dict[str, str]:
    constraints = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.count("==") != 1:
            raise BaselineError(
                "constraint must use exactly one == pin: {0}".format(stripped)
            )
        package_name, version = stripped.split("==", 1)
        normalized_name = package_name.strip().lower().replace("_", "-")
        _validate_version(version, normalized_name)
        constraints[normalized_name] = version
    return constraints
