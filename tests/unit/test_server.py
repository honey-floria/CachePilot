from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cachepilot import server
from cachepilot.gateway.backends import DeterministicChatBackend


def cuda_runtime():
    torch = MagicMock()
    torch.__version__ = "2.11.0"
    torch.version.cuda = "12.8"
    torch.cuda.is_available.return_value = True
    torch.cuda.device_count.return_value = 2
    torch.cuda.is_bf16_supported.return_value = True
    torch.cuda.get_device_properties.return_value = SimpleNamespace(
        name="Test NVIDIA GPU", total_memory=8 * 1024**3
    )
    return torch


@pytest.mark.parametrize("device_index", [-1, 2])
def test_invalid_device_fails_before_loading_model(device_index):
    with patch.dict("sys.modules", {"torch": cuda_runtime()}):
        with pytest.raises(RuntimeError, match="GPU 索引"):
            server.check_cuda(device_index, "bfloat16")


def test_cpu_only_torch_is_rejected():
    torch = cuda_runtime()
    torch.version.cuda = None
    with patch.dict("sys.modules", {"torch": torch}):
        with pytest.raises(RuntimeError, match="CUDA 不可用"):
            server.check_cuda(0, "bfloat16")


def test_unsupported_bfloat16_is_rejected():
    torch = cuda_runtime()
    torch.cuda.is_bf16_supported.return_value = False
    with patch.dict("sys.modules", {"torch": torch}):
        with pytest.raises(RuntimeError, match="--dtype float16"):
            server.check_cuda(0, "bfloat16")
    torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)


def test_cuda_kernel_failure_is_reported():
    torch = cuda_runtime()
    torch.ones.side_effect = RuntimeError("no kernel image")
    with patch.dict("sys.modules", {"torch": torch}):
        with pytest.raises(RuntimeError, match="CUDA 运算检查失败"):
            server.check_cuda(0, "float16")


def test_check_only_does_not_load_model_or_start_server():
    torch = cuda_runtime()
    with (
        patch.dict("sys.modules", {"torch": torch}),
        patch.object(server, "TorchExecutor") as executor,
        patch.object(server.uvicorn, "run") as run,
    ):
        server.main(["--check", "--device", "1", "--dtype", "float16"])
    torch.cuda.set_device.assert_called_once_with(1)
    torch.ones.assert_called_once_with(1, device="cuda:1", dtype=torch.float16)
    executor.assert_not_called()
    run.assert_not_called()


def test_gpu_app_uses_real_executor_and_its_tokenizer():
    backend = DeterministicChatBackend()
    backend.count_prompt_tokens = lambda request: 8
    with (
        patch.object(server, "check_cuda", return_value="cuda:1"),
        patch.object(server, "TorchExecutor", return_value=backend) as executor,
    ):
        app = server.create_gpu_app(device_index=1, dtype="float16")
    config = executor.call_args.args[0]
    assert config.device == "cuda:1"
    assert config.dtype == "float16"
    assert len(config.model_revision) == 40
    runtime = app.state.gateway_runtime
    assert runtime.backend is backend
    assert runtime.token_counter is backend
    assert runtime.settings.max_active_sequences == 8
    assert runtime.settings.execution_slots == 1
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        response = client.post(
            "/v1/chat/completions",
            headers={"X-Tenant-ID": "team-a"},
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
                "stream": False,
            },
        )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 8


def test_cuda_failure_exits_without_starting_http_server():
    with (
        patch.object(server, "check_cuda", side_effect=RuntimeError("CUDA unavailable")),
        patch.object(server.uvicorn, "run") as run,
        pytest.raises(SystemExit) as result,
    ):
        server.main([])
    assert result.value.code == 1
    run.assert_not_called()
