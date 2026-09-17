import importlib
import unittest


class PackageImportTests(unittest.TestCase):
    def test_skeleton_packages_import(self):
        for package in (
            "cachepilot.cache",
            "cachepilot.config",
            "cachepilot.executors",
            "cachepilot.gateway",
            "cachepilot.routing",
            "cachepilot.runtime",
            "cachepilot.telemetry",
        ):
            with self.subTest(package=package):
                importlib.import_module(package)
