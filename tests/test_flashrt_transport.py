"""Transport tests. Only the `local` transport actually runs anything; it shells out
to this interpreter, so there is still no GPU, no network and no FlashRT install.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.tasks.flashrt.transport import (  # noqa: E402
    DEFAULT_REFEREE_CMD,
    SOURCES_DIR_TOKEN,
    LocalTransport,
    ServiceTransport,
    SshTransport,
    TransportResult,
    make_transport,
)


# A stand-in referee: echoes the argv it was given plus the files it can see.
FAKE_REFEREE = """
import json, os, sys
argv = sys.argv[1:]
src = ""
if "--sources" in argv:
    src = argv[argv.index("--sources") + 1]
files = sorted(os.listdir(src)) if src and os.path.isdir(src) else []
print(json.dumps({"argv": argv, "files": files, "ok": True}))
"""


class LocalTransportTests(unittest.TestCase):
    def _transport(self, script: str = FAKE_REFEREE, **kwargs) -> LocalTransport:
        return LocalTransport(referee_cmd=[sys.executable, "-c", script], **kwargs)

    def test_sources_are_materialized_and_the_token_substituted(self) -> None:
        res = self._transport().run(
            ["evaluate", "t1", "--sources", SOURCES_DIR_TOKEN, "--json"],
            sources={"kernel.h": "#pragma once", "kernel.cu": "__global__ void k(){}"},
            timeout_seconds=30,
        )
        self.assertEqual(res.returncode, 0)
        self.assertTrue(res.ok())
        payload = json.loads(res.stdout)
        self.assertEqual(payload["files"], ["kernel.cu", "kernel.h"])
        self.assertNotIn(SOURCES_DIR_TOKEN, payload["argv"])
        self.assertEqual(payload["argv"][0], "evaluate")

    def test_sources_dir_is_cleaned_up(self) -> None:
        res = self._transport().run(
            ["evaluate", "t1", "--sources", SOURCES_DIR_TOKEN],
            sources={"kernel.cu": "x"},
            timeout_seconds=30,
        )
        src_dir = json.loads(res.stdout)["argv"][3]
        self.assertFalse(Path(src_dir).exists())

    def test_no_sources_means_no_directory(self) -> None:
        res = self._transport().run(["list"], timeout_seconds=30)
        self.assertEqual(json.loads(res.stdout)["argv"], ["list"])

    def test_nonzero_exit_is_reported_not_raised(self) -> None:
        res = self._transport(script="import sys; sys.stderr.write('kaboom'); sys.exit(3)").run(
            ["evaluate", "t1"], timeout_seconds=30
        )
        self.assertEqual(res.returncode, 3)
        self.assertFalse(res.ok())
        self.assertIn("kaboom", res.stderr)
        self.assertIn("kaboom", res.combined_log())

    def test_timeout_is_reported_not_raised(self) -> None:
        res = self._transport(script="import time; time.sleep(30)").run(["evaluate", "t1"], timeout_seconds=0.5)
        self.assertTrue(res.timed_out)
        self.assertEqual(res.returncode, 124)
        self.assertFalse(res.ok())
        self.assertIn("timed out", res.stderr)

    def test_missing_referee_binary_is_reported_not_raised(self) -> None:
        res = LocalTransport(referee_cmd="definitely-not-a-real-binary-xyz list").run(["list"], timeout_seconds=5)
        self.assertEqual(res.returncode, 127)
        self.assertIn("not found", res.stderr)

    def test_describe_for_logging_is_json_friendly(self) -> None:
        json.dumps(LocalTransport().describe_for_logging())
        self.assertEqual(LocalTransport().describe_for_logging()["referee_cmd"], DEFAULT_REFEREE_CMD)


class TransportFactoryTests(unittest.TestCase):
    def test_local(self) -> None:
        self.assertIsInstance(make_transport("local"), LocalTransport)

    def test_ssh_requires_a_host(self) -> None:
        self.assertIsInstance(make_transport("ssh", host="thor-01"), SshTransport)
        with self.assertRaises(ValueError):
            make_transport("ssh")

    def test_service_requires_an_endpoint(self) -> None:
        self.assertIsInstance(make_transport("service", endpoint="http://x/referee"), ServiceTransport)
        with self.assertRaises(ValueError):
            make_transport("service")

    def test_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            make_transport("carrier-pigeon")

    def test_ssh_and_service_describe_for_logging(self) -> None:
        json.dumps(SshTransport(host="thor-01").describe_for_logging())
        json.dumps(ServiceTransport(endpoint="http://x").describe_for_logging())

    def test_empty_endpoints_fail_at_run_time_without_touching_the_network(self) -> None:
        self.assertEqual(SshTransport(host="").run(["list"]).returncode, 2)
        self.assertEqual(ServiceTransport(endpoint="").run(["list"]).returncode, 2)


class TransportResultTests(unittest.TestCase):
    def test_combined_log_truncates(self) -> None:
        res = TransportResult(returncode=0, stdout="a" * 100, stderr="b" * 100)
        self.assertTrue(res.combined_log(max_chars=10).startswith("aaaaaaaaaa"))
        self.assertIn("truncated", res.combined_log(max_chars=10))


if __name__ == "__main__":
    unittest.main()
