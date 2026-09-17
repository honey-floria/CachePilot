import json
import tempfile
import unittest
from pathlib import Path

from cachepilot.config.baseline import (
    BaselineError,
    load_dependency_baseline,
    load_model_baseline,
    validate_repository_baselines,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPOSITORY_ROOT / "config" / "model.json"
DEPENDENCY_CONFIG = REPOSITORY_ROOT / "config" / "dependencies.json"


class ModelBaselineTests(unittest.TestCase):
    def test_model_and_tokenizer_are_pinned_to_the_same_commit(self):
        baseline = load_model_baseline(MODEL_CONFIG)

        self.assertEqual("Qwen/Qwen2.5-0.5B-Instruct", baseline.model_id)
        self.assertEqual(baseline.model_id, baseline.tokenizer_id)
        self.assertEqual(
            "7ae557604adf67be50417f59c2c2f167def9a775",
            baseline.model_revision,
        )
        self.assertEqual(baseline.model_revision, baseline.tokenizer_revision)
        self.assertFalse(baseline.trust_remote_code)

    def test_license_context_and_architecture_are_recorded(self):
        baseline = load_model_baseline(MODEL_CONFIG)

        self.assertEqual("Apache-2.0", baseline.license_spdx_id)
        self.assertEqual(32768, baseline.model_max_context_tokens)
        self.assertEqual(8192, baseline.service_context_limit)
        self.assertEqual("qwen2", baseline.model_type)
        self.assertEqual(24, baseline.num_hidden_layers)
        self.assertEqual(2, baseline.num_key_value_heads)
        self.assertEqual(64, baseline.head_dim)

    def test_model_sources_use_the_pinned_revision(self):
        payload = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
        revision = payload["model_revision"]

        self.assertIn(revision, payload["license"]["source_url"])
        self.assertIn(revision, payload["sources"]["model_config"])
        self.assertNotIn("/main/", payload["license"]["source_url"])
        self.assertNotIn("/main/", payload["sources"]["model_config"])

    def test_floating_revision_is_rejected(self):
        payload = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
        payload["model_revision"] = "main"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BaselineError):
                load_model_baseline(path)


class DependencyBaselineTests(unittest.TestCase):
    def test_python_and_gpu_stack_are_exactly_pinned(self):
        baseline = load_dependency_baseline(DEPENDENCY_CONFIG)
        profiles = {profile.name: profile for profile in baseline.profiles}
        vllm = profiles["vllm_executor"]

        self.assertEqual("3.13.15", baseline.python_version)
        self.assertEqual("2.11.0", vllm.torch)
        self.assertEqual("5.5.3", vllm.transformers)
        self.assertEqual("0.24.0", vllm.vllm)
        self.assertIn("GPU validation pending", vllm.status)

    def test_torch_and_vllm_profiles_share_model_stack_versions(self):
        baseline = load_dependency_baseline(DEPENDENCY_CONFIG)
        profiles = {profile.name: dict(profile.packages) for profile in baseline.profiles}

        for package in (
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface_hub",
        ):
            self.assertEqual(
                profiles["torch_executor"][package],
                profiles["vllm_executor"][package],
            )

    def test_constraints_and_python_version_match_machine_readable_config(self):
        validate_repository_baselines(REPOSITORY_ROOT)

    def test_no_selected_version_uses_a_range_or_floating_name(self):
        baseline = load_dependency_baseline(DEPENDENCY_CONFIG)
        for profile in baseline.profiles:
            self.assertRegex(profile.python, r"^\d+\.\d+\.\d+$")
            for package, version in profile.packages:
                with self.subTest(profile=profile.name, package=package):
                    self.assertRegex(version, r"^\d+\.\d+\.\d+$")
                    self.assertNotIn(version.lower(), {"latest", "main", "nightly"})


class OpenAPIModelPinTests(unittest.TestCase):
    def test_openapi_accepts_only_the_pinned_model_id(self):
        openapi = json.loads(
            (REPOSITORY_ROOT / "contracts" / "openapi.json").read_text(
                encoding="utf-8"
            )
        )
        model_schema = openapi["components"]["schemas"]["ChatCompletionRequest"][
            "properties"
        ]["model"]

        self.assertEqual("Qwen/Qwen2.5-0.5B-Instruct", model_schema["const"])


if __name__ == "__main__":
    unittest.main()
