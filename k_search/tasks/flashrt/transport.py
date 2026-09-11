"""How the FlashRT referee is invoked.

The referee is a CLI on the validation device:

    <referee_cmd> list
    <referee_cmd> describe <tunable_id>
    <referee_cmd> evaluate <tunable_id> --sources <dir> --json

K-Search never imports FlashRT; it shells out. That process boundary is what lets
the search host and the validation device be different machines, so this module is
the *only* place that knows how the two are connected.

Three transports implement one interface:

- `local`   — subprocess on this host (complete)
- `ssh`     — copy the sources over, run the CLI remotely, clean up (thin but real)
- `service` — HTTP POST to a referee service that does the same (thin but real)

Transports return raw stdout/stderr and an exit code. Parsing the JSON contract is
the caller's job: a transport must not care what the referee said.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


# Placeholder token used in an argv to mean "the directory holding the candidate
# sources, wherever the transport put it". The transport substitutes the real path.
SOURCES_DIR_TOKEN = "{sources}"

DEFAULT_REFEREE_CMD = "python -m flash_rt.kopt.referee"
DEFAULT_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class TransportResult:
    """Raw outcome of one referee invocation. No parsing, no interpretation."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    command: str = ""

    def ok(self) -> bool:
        return int(self.returncode) == 0 and not self.timed_out

    def combined_log(self, *, max_chars: int = 0) -> str:
        parts = [p for p in (self.stdout.strip(), self.stderr.strip()) if p]
        s = "\n".join(parts)
        if max_chars and len(s) > int(max_chars):
            s = s[: int(max_chars)] + "...<truncated>..."
        return s


class Transport(Protocol):
    """One referee invocation, wherever the referee lives."""

    kind: str

    def run(
        self,
        argv: Sequence[str],
        *,
        sources: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResult: ...

    def describe_for_logging(self) -> dict[str, Any]: ...


def _split_referee_cmd(referee_cmd: str | Sequence[str]) -> list[str]:
    if isinstance(referee_cmd, str):
        return shlex.split(referee_cmd)
    return [str(a) for a in referee_cmd]


def _materialize_sources(sources: Mapping[str, str], dest: Path) -> None:
    """Write the candidate's source files into `dest` (flat; no subdirectories)."""
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in (sources or {}).items():
        name = os.path.basename(str(rel))
        if not name:
            continue
        (dest / name).write_text(str(content or ""), encoding="utf-8")


def _substitute(argv: Sequence[str], sources_dir: str) -> list[str]:
    return [str(a).replace(SOURCES_DIR_TOKEN, sources_dir) for a in argv]


@dataclass
class LocalTransport:
    """Run the referee as a subprocess on this host."""

    referee_cmd: str | Sequence[str] = DEFAULT_REFEREE_CMD
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    keep_tmp: bool = False
    kind: str = "local"

    def run(
        self,
        argv: Sequence[str],
        *,
        sources: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResult:
        base = _split_referee_cmd(self.referee_cmd)
        tmpdir: str | None = None
        try:
            if sources:
                tmpdir = tempfile.mkdtemp(prefix="ksearch_flashrt_src_")
                _materialize_sources(sources, Path(tmpdir))
            full = base + _substitute(argv, tmpdir or "")
            cmd_text = " ".join(shlex.quote(a) for a in full)

            run_env = dict(os.environ)
            run_env.update({str(k): str(v) for k, v in (self.env or {}).items()})

            try:
                proc = subprocess.run(
                    full,
                    capture_output=True,
                    text=True,
                    timeout=(float(timeout_seconds) if timeout_seconds else DEFAULT_TIMEOUT_SECONDS),
                    cwd=(str(self.cwd) if self.cwd else None),
                    env=run_env,
                )
            except subprocess.TimeoutExpired as e:
                return TransportResult(
                    returncode=124,
                    stdout=_as_text(e.stdout),
                    stderr=(_as_text(e.stderr) + f"\n[transport] timed out after {timeout_seconds}s").strip(),
                    timed_out=True,
                    command=cmd_text,
                )
            except FileNotFoundError as e:
                return TransportResult(
                    returncode=127,
                    stderr=f"[transport] referee command not found: {e}",
                    command=cmd_text,
                )
            except Exception as e:  # pragma: no cover - defensive
                return TransportResult(
                    returncode=1,
                    stderr=f"[transport] {type(e).__name__}: {e}",
                    command=cmd_text,
                )

            return TransportResult(
                returncode=int(proc.returncode),
                stdout=str(proc.stdout or ""),
                stderr=str(proc.stderr or ""),
                command=cmd_text,
            )
        finally:
            if tmpdir and not self.keep_tmp:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def describe_for_logging(self) -> dict[str, Any]:
        return {
            "transport": self.kind,
            "referee_cmd": " ".join(_split_referee_cmd(self.referee_cmd)),
            "cwd": (str(self.cwd) if self.cwd else None),
        }


@dataclass
class SshTransport:
    """Copy the sources to the validation device, run the referee there, clean up.

    Deliberately thin: it shells out to the system `ssh` / `scp`, so authentication,
    multiplexing and host config all come from the user's ssh setup rather than
    being re-implemented here.
    """

    host: str
    referee_cmd: str | Sequence[str] = DEFAULT_REFEREE_CMD
    remote_tmp_root: str = "/tmp"
    ssh_bin: str = "ssh"
    scp_bin: str = "scp"
    ssh_opts: Sequence[str] = ()
    kind: str = "ssh"

    def run(
        self,
        argv: Sequence[str],
        *,
        sources: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResult:
        if not str(self.host or "").strip():
            return TransportResult(returncode=2, stderr="[transport] ssh transport needs a host")

        timeout = float(timeout_seconds) if timeout_seconds else DEFAULT_TIMEOUT_SECONDS
        opts = [str(o) for o in (self.ssh_opts or ())]
        remote_dir = ""
        local_tmp: str | None = None
        try:
            if sources:
                remote_dir = f"{self.remote_tmp_root.rstrip('/')}/ksearch_flashrt_{uuid.uuid4().hex[:12]}"
                local_tmp = tempfile.mkdtemp(prefix="ksearch_flashrt_src_")
                _materialize_sources(sources, Path(local_tmp))

                mk = self._ssh(["mkdir", "-p", remote_dir], opts=opts, timeout=timeout)
                if not mk.ok():
                    return mk
                files = [str(p) for p in sorted(Path(local_tmp).iterdir())]
                cp = _run_local(
                    [self.scp_bin, *opts, *files, f"{self.host}:{remote_dir}/"], timeout=timeout
                )
                if not cp.ok():
                    return cp

            remote_argv = _split_referee_cmd(self.referee_cmd) + _substitute(argv, remote_dir)
            return self._ssh(remote_argv, opts=opts, timeout=timeout)
        finally:
            if local_tmp:
                shutil.rmtree(local_tmp, ignore_errors=True)
            if remote_dir:
                try:
                    self._ssh(["rm", "-rf", remote_dir], opts=opts, timeout=60.0)
                except Exception:
                    pass

    def _ssh(self, remote_argv: Sequence[str], *, opts: Sequence[str], timeout: float) -> TransportResult:
        remote = " ".join(shlex.quote(str(a)) for a in remote_argv)
        return _run_local([self.ssh_bin, *opts, self.host, remote], timeout=timeout)

    def describe_for_logging(self) -> dict[str, Any]:
        return {
            "transport": self.kind,
            "host": str(self.host),
            "referee_cmd": " ".join(_split_referee_cmd(self.referee_cmd)),
            "remote_tmp_root": str(self.remote_tmp_root),
        }


@dataclass
class ServiceTransport:
    """POST the invocation to a referee HTTP service that fronts the device.

    Request body:  {"argv": [...], "sources": {"kernel.cu": "..."}, "timeout_seconds": N}
    Response body: {"returncode": 0, "stdout": "...", "stderr": "..."}

    Uses `urllib` so the adapter keeps no HTTP dependency.
    """

    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    kind: str = "service"

    def run(
        self,
        argv: Sequence[str],
        *,
        sources: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResult:
        if not str(self.endpoint or "").strip():
            return TransportResult(returncode=2, stderr="[transport] service transport needs an endpoint")

        timeout = float(timeout_seconds) if timeout_seconds else DEFAULT_TIMEOUT_SECONDS
        payload = {
            "argv": [str(a) for a in argv],
            "sources": {str(k): str(v) for k, v in (sources or {}).items()},
            "sources_dir_token": SOURCES_DIR_TOKEN,
            "timeout_seconds": timeout,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            str(self.endpoint),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", **{str(k): str(v) for k, v in (self.headers or {}).items()}},
        )
        cmd_text = f"POST {self.endpoint} {' '.join(str(a) for a in argv)}"
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return TransportResult(
                returncode=int(e.code),
                stderr=f"[transport] HTTP {e.code}: {e.reason}",
                command=cmd_text,
            )
        except TimeoutError:
            return TransportResult(
                returncode=124,
                stderr=f"[transport] timed out after {timeout}s",
                timed_out=True,
                command=cmd_text,
            )
        except Exception as e:
            return TransportResult(returncode=1, stderr=f"[transport] {type(e).__name__}: {e}", command=cmd_text)

        try:
            obj = json.loads(raw)
        except Exception:
            # A service that streams the referee's stdout verbatim is still usable.
            return TransportResult(returncode=0, stdout=raw, command=cmd_text)
        if not isinstance(obj, dict):
            return TransportResult(returncode=0, stdout=raw, command=cmd_text)
        return TransportResult(
            returncode=int(obj.get("returncode", 0) or 0),
            stdout=str(obj.get("stdout", "") or ""),
            stderr=str(obj.get("stderr", "") or ""),
            timed_out=bool(obj.get("timed_out", False)),
            command=cmd_text,
        )

    def describe_for_logging(self) -> dict[str, Any]:
        return {"transport": self.kind, "endpoint": str(self.endpoint)}


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _run_local(cmd: Sequence[str], *, timeout: float) -> TransportResult:
    cmd_text = " ".join(shlex.quote(str(a)) for a in cmd)
    try:
        proc = subprocess.run([str(a) for a in cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return TransportResult(
            returncode=124,
            stdout=_as_text(e.stdout),
            stderr=(_as_text(e.stderr) + f"\n[transport] timed out after {timeout}s").strip(),
            timed_out=True,
            command=cmd_text,
        )
    except FileNotFoundError as e:
        return TransportResult(returncode=127, stderr=f"[transport] command not found: {e}", command=cmd_text)
    except Exception as e:  # pragma: no cover - defensive
        return TransportResult(returncode=1, stderr=f"[transport] {type(e).__name__}: {e}", command=cmd_text)
    return TransportResult(
        returncode=int(proc.returncode),
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
        command=cmd_text,
    )


def make_transport(
    kind: str = "local",
    *,
    referee_cmd: str | Sequence[str] = DEFAULT_REFEREE_CMD,
    host: str = "",
    endpoint: str = "",
    **kwargs: Any,
) -> Transport:
    """Build a transport by name. `host` is required for ssh, `endpoint` for service."""
    k = str(kind or "local").strip().lower()
    if k == "local":
        return LocalTransport(referee_cmd=referee_cmd, **kwargs)
    if k == "ssh":
        if not str(host or "").strip():
            raise ValueError("--transport ssh requires a host (e.g. --flashrt-referee-host thor-01)")
        return SshTransport(host=str(host), referee_cmd=referee_cmd, **kwargs)
    if k == "service":
        if not str(endpoint or "").strip():
            raise ValueError("--transport service requires an endpoint URL")
        return ServiceTransport(endpoint=str(endpoint), **kwargs)
    raise ValueError(f"Unsupported transport: {kind!r} (expected local | ssh | service)")
