"""依赖标准库的空 HTTP 服务。

它只用于仓库骨架和环境验收，不执行模型推理，也不宣称实现 OpenAI API。
服务提供健康检查、就绪检查和最小 metrics 端点，便于本地与 Colab smoke test。
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping


class EmptyServiceHandler(BaseHTTPRequestHandler):
    """返回稳定 JSON 的最小探活处理器。"""

    server_version = "CachePilotEmptyService/0.1"

    def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok", "service": "cachepilot"})
            return
        if self.path == "/readyz":
            self._write_json(200, {"status": "ready", "service": "cachepilot"})
            return
        if self.path == "/metrics":
            body = "cachepilot_empty_service_up 1\n"
            encoded = body.encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self._write_json(404, {"error": {"code": "not_found"}})

    def log_message(self, format: str, *args: object) -> None:
        # Keep Colab output concise while retaining the standard server hook.
        return


def create_server(
    host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    """创建但不启动空服务，便于测试和嵌入式 smoke test。"""

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return ThreadingHTTPServer((host, port), EmptyServiceHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("CACHEPILOT_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
    )
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(
        f"CachePilot empty service listening on http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
