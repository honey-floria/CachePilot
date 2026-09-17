"""运行时骨架：当前提供可探活的空服务，后续承载 Registry/Worker loop。"""

from .empty_service import create_server

__all__ = ["create_server"]
