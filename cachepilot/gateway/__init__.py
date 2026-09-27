"""网关契约、FastAPI 服务与可替换生成后端。"""

from .backends import (
    ChatBackend,
    ConservativePromptTokenCounter,
    DeterministicChatBackend,
    GeneratedText,
    PromptTokenCounter,
)

__all__ = [
    "ChatBackend",
    "ConservativePromptTokenCounter",
    "DeterministicChatBackend",
    "GatewayRuntime",
    "GatewaySettings",
    "GeneratedText",
    "PromptTokenCounter",
    "create_app",
]


def __getattr__(name):
    """Load FastAPI objects only when requested to avoid runtime import cycles."""

    if name in {"GatewayRuntime", "GatewaySettings", "create_app"}:
        from .api import GatewayRuntime, GatewaySettings, create_app

        return {
            "GatewayRuntime": GatewayRuntime,
            "GatewaySettings": GatewaySettings,
            "create_app": create_app,
        }[name]
    raise AttributeError(name)
