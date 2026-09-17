import errno
import json
import threading
import unittest
from urllib.request import urlopen

from cachepilot.runtime.empty_service import create_server


class EmptyServiceTests(unittest.TestCase):
    def test_empty_service_health_ready_and_metrics(self):
        try:
            server = create_server("127.0.0.1", 0)
        except PermissionError as exc:
            if exc.errno == errno.EPERM:
                self.skipTest("sandbox does not permit local socket binding")
            raise
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/healthz") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["status"], "ok")
            with urlopen(base + "/readyz") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["status"], "ready")
            with urlopen(base + "/metrics") as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"cachepilot_empty_service_up 1", response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
