# -*- coding: utf-8 -*-
"""
Launcher for the campus twin demo.

It prefers port 5006, but it only reuses an existing service when that service
is confirmed to be the original campus demo. If another Panel app is occupying
the port, the launcher will start the demo on the next clean fallback port.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge_startup import print_hook_summary, run_startup_hooks
from src.mcp_profile_menu import apply_mcp_profile_menu_if_requested

PREFERRED_PORT = 5006
FALLBACK_PORTS = (5008, 5009, 5010, 5011)
WAIT_SECONDS = 90
START_RETRIES = 1
PORT_CHECK_INTERVAL = 0.5
HTTP_PROBE_TIMEOUT = 8.0
HTTP_RETRY_INTERVAL = 1.0
PANEL_LOG_DIR = ROOT / "dev_artifacts" / "logs"

DEMO_MARKERS = (
    "校園能源數位分身",
    "Energy Digital Twin",
    "Control Room",
)

WORKBENCH_MARKERS = (
    "Building Energy Knowledge Workbench",
    "Knowledge Base Status",
    "Upload & Status",
)


@dataclass(frozen=True)
class LauncherConfig:
    preferred_port: int
    fallback_ports: tuple[int, ...]
    wait_seconds: int
    start_retries: int


def _app_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/app"


def _valid_port(port: int) -> bool:
    return 1 <= port <= 65535


def _env_int(names: tuple[str, ...], default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            print(f"[launcher config] ignoring invalid {name}={raw!r}; using {default}.")
            return default
        if min_value is not None and value < min_value:
            print(f"[launcher config] ignoring {name}={value}; minimum is {min_value}.")
            return default
        if max_value is not None and value > max_value:
            print(f"[launcher config] ignoring {name}={value}; maximum is {max_value}.")
            return default
        return value
    return default


def _parse_port_list(raw: str) -> tuple[int, ...]:
    ports: list[int] = []
    seen: set[int] = set()
    for item in raw.replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        try:
            port = int(text)
        except ValueError:
            print(f"[launcher config] ignoring invalid fallback port {text!r}.")
            continue
        if not _valid_port(port):
            print(f"[launcher config] ignoring out-of-range fallback port {port}.")
            continue
        if port in seen:
            continue
        ports.append(port)
        seen.add(port)
    return tuple(ports)


def _load_config() -> LauncherConfig:
    preferred_port = _env_int(("ENERGY_DEMO_PORT", "PORT"), PREFERRED_PORT, min_value=1, max_value=65535)

    fallback_raw = os.getenv("ENERGY_DEMO_FALLBACK_PORTS", "").strip()
    if fallback_raw:
        fallback_ports = _parse_port_list(fallback_raw)
    else:
        fallback_ports = tuple(p for p in (PREFERRED_PORT, *FALLBACK_PORTS) if p != preferred_port)
    fallback_ports = tuple(p for p in fallback_ports if p != preferred_port)
    if not fallback_ports:
        fallback_ports = tuple(p for p in FALLBACK_PORTS if p != preferred_port)

    return LauncherConfig(
        preferred_port=preferred_port,
        fallback_ports=fallback_ports,
        wait_seconds=_env_int(("ENERGY_DEMO_WAIT_SECONDS",), WAIT_SECONDS, min_value=5),
        start_retries=_env_int(("ENERGY_DEMO_START_RETRIES",), START_RETRIES, min_value=0),
    )


def _panel_log_path(port: int) -> Path:
    configured = os.getenv("ENERGY_DEMO_PANEL_LOG_DIR", "").strip()
    log_dir = Path(configured).expanduser() if configured else PANEL_LOG_DIR
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"panel_server_{port}.log"


def _fetch_text(url: str, timeout: float = 2.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="ignore")
    except Exception:
        return None


def _classify_app_html(html: str | None) -> str | None:
    if not html:
        return None
    if any(marker in html for marker in DEMO_MARKERS):
        return "demo"
    if any(marker in html for marker in WORKBENCH_MARKERS):
        return "workbench"
    return "other"


def _probe_app(port: int, timeout: float = 2.0) -> str | None:
    return _classify_app_html(_fetch_text(_app_url(port), timeout=timeout))


def _probe_websocket_allowed(port: int, timeout: float = 3.0) -> bool:
    """Return True only if the server accepts WebSocket connections from 127.0.0.1.

    A server started without --allow-websocket-origin=127.0.0.1:<port> will
    respond with HTTP 403 to the upgrade handshake, causing the dashboard to
    render blank (HTML loads but Panel cannot communicate).
    """
    import http.client
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request(
            "GET",
            "/app/ws",
            headers={
                "Host": f"127.0.0.1:{port}",
                "Origin": f"http://127.0.0.1:{port}",
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
                "Sec-WebSocket-Version": "13",
            },
        )
        resp = conn.getresponse()
        conn.close()
        # 101 = switched, 200/4xx other → 403 means origin rejected
        return resp.status != 403
    except Exception:
        # Connection refused or timeout → server not ready; treat as "unknown"
        return True


def _probe_http_ready(
    port: int,
    timeout: float = 2.0,
    fetch_text_fn=None,
) -> bool:
    fetch_text = fetch_text_fn or _fetch_text
    return fetch_text(_app_url(port), timeout=timeout) is not None


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _listening_pid(port: int) -> int | None:
    command = (
        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
        "| Select-Object -First 1 -ExpandProperty OwningProcess"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text.isdigit():
        return None
    return int(text)


def _process_command_line(pid: int) -> str:
    command = (
        f"Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\" "
        "| Select-Object -ExpandProperty CommandLine"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _reset_stale_demo_server(port: int) -> bool:
    pid = _listening_pid(port)
    if pid is None:
        return False

    command_line = _process_command_line(pid).lower()
    if "panel" not in command_line or "serve" not in command_line or "app.py" not in command_line:
        return False

    stop_command = f"Stop-Process -Id {pid} -Force"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", stop_command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False

    for _ in range(10):
        time.sleep(0.5)
        if not _port_in_use(port):
            return True
    return False


def _choose_target_port(
    *,
    preferred_port: int = PREFERRED_PORT,
    fallback_ports: tuple[int, ...] = FALLBACK_PORTS,
    excluded_ports: set[int] | None = None,
    probe_app_fn=None,
    port_in_use_fn=None,
    reset_port_fn=None,
) -> tuple[int, bool]:
    probe = probe_app_fn or _probe_app
    port_in_use = port_in_use_fn or _port_in_use
    reset_port = reset_port_fn or _reset_stale_demo_server
    excluded = excluded_ports or set()

    if preferred_port not in excluded:
        preferred_kind = probe(preferred_port)
        if preferred_kind == "demo":
            # Verify the running demo accepts WebSocket from 127.0.0.1.
            # A server started without --allow-websocket-origin will serve HTML but
            # refuse the WebSocket upgrade, leaving the dashboard completely blank.
            if _probe_websocket_allowed(preferred_port):
                return preferred_port, False
            # WebSocket origin rejected → kill and restart with correct flags.
            reset_port(preferred_port)
            return preferred_port, True
        if preferred_kind is None and not port_in_use(preferred_port):
            return preferred_port, True
        if preferred_kind in {"workbench", "other"} and port_in_use(preferred_port) and reset_port(preferred_port):
            return preferred_port, True

    for port in fallback_ports:
        if port in excluded:
            continue
        kind = probe(port)
        if kind == "demo":
            return port, False
        if kind is None and not port_in_use(port):
            return port, True

    raise RuntimeError(
        "No clean port is available for the campus demo. "
        f"Tried {preferred_port} and {', '.join(str(port) for port in fallback_ports)}."
    )


def _start_demo_server(port: int, *, log_path: Path | None = None) -> subprocess.Popen:
    log_path = log_path or _panel_log_path(port)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"cwd={ROOT}\n")
        log_file.write(f"port={port}\n")
        log_file.write("cmd=python -m panel serve app.py\n\n")
        log_file.flush()
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "panel",
                "serve",
                "app.py",
                "--address",
                "127.0.0.1",
                "--port",
                str(port),
                "--allow-websocket-origin",
                f"127.0.0.1:{port}",
                "--allow-websocket-origin",
                f"localhost:{port}",
            ],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def _early_exit_error(process: subprocess.Popen | None, port: int, log_path: Path | None) -> str | None:
    if process is None:
        return None
    exit_code = process.poll()
    if exit_code is None:
        return None
    suffix = f" See Panel log: {log_path}" if log_path is not None else ""
    return f"Panel server on port {port} exited before it became ready (exit code {exit_code}).{suffix}"


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)


def _wait_for_http_ready(
    port: int,
    timeout_seconds: int = WAIT_SECONDS,
    *,
    process: subprocess.Popen | None = None,
    log_path: Path | None = None,
    port_check_interval: float = PORT_CHECK_INTERVAL,
    http_probe_timeout: float = HTTP_PROBE_TIMEOUT,
    http_retry_interval: float = HTTP_RETRY_INTERVAL,
    port_in_use_fn=None,
    probe_http_ready_fn=None,
    sleep_fn=None,
    time_fn=None,
) -> tuple[bool, str | None]:
    port_in_use = port_in_use_fn or _port_in_use
    probe_http_ready = probe_http_ready_fn or _probe_http_ready
    sleep = sleep_fn or time.sleep
    now = time_fn or time.monotonic

    deadline = now() + float(timeout_seconds)
    while now() < deadline:
        error = _early_exit_error(process, port, log_path)
        if error:
            return False, error
        if port_in_use(port):
            break
        sleep(port_check_interval)
    else:
        return False, f"Port {port} did not start listening within {timeout_seconds} seconds."

    while now() < deadline:
        error = _early_exit_error(process, port, log_path)
        if error:
            return False, error
        if probe_http_ready(port, timeout=http_probe_timeout):
            return True, None
        sleep(http_retry_interval)

    return False, f"Port {port} started listening, but {_app_url(port)} did not become ready within {timeout_seconds} seconds."


def _open_url(url: str) -> bool:
    try:
        if webbrowser.open(url):
            return True
    except Exception:
        pass

    try:
        if sys.platform.startswith("win"):
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        return True
    except Exception:
        return False


def main() -> None:
    config = _load_config()
    print(
        "[launcher config] preferred_port="
        f"{config.preferred_port}, fallback_ports={','.join(str(p) for p in config.fallback_ports) or '(none)'}, "
        f"wait_seconds={config.wait_seconds}, start_retries={config.start_retries}"
    )

    hooks_ran = False
    failed_ports: set[int] = set()
    last_error: str | None = None
    max_attempts = config.start_retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            port, should_start = _choose_target_port(
                preferred_port=config.preferred_port,
                fallback_ports=config.fallback_ports,
                excluded_ports=failed_ports,
            )
        except RuntimeError as exc:
            if last_error:
                raise RuntimeError(f"{last_error}\n{exc}") from exc
            raise

        url = _app_url(port)
        if not should_start:
            print(f"Reusing the running campus demo on port {port}.")
            print(f"Opening {url}")
            if not _open_url(url):
                print(f"Automatic browser launch failed. Open this URL manually: {url}")
            return

        try:
            if not hooks_ran:
                menu_info = apply_mcp_profile_menu_if_requested(ROOT)
                if menu_info.get("applied"):
                    print(f"[mcp menu] applied profiles from {menu_info.get('path')}")
                hook_results = run_startup_hooks(ROOT)
                print_hook_summary(hook_results)
                hooks_ran = True
        except Exception as exc:
            print(f"[startup hooks] FATAL: {exc}")
            raise

        log_path = _panel_log_path(port)
        if attempt > 1:
            print(f"Retrying campus demo startup (attempt {attempt}/{max_attempts})...")
        print(f"Starting campus demo on port {port}...")
        print(f"Panel log: {log_path}")
        process = _start_demo_server(port, log_path=log_path)
        print(f"Waiting for the demo to be ready at {url}...")
        ready, error = _wait_for_http_ready(
            port,
            timeout_seconds=config.wait_seconds,
            process=process,
            log_path=log_path,
        )
        if not ready:
            last_error = error or f"The campus demo did not become ready at {url}"
            print(f"[launcher retry] attempt {attempt}/{max_attempts} failed: {last_error}")
            failed_ports.add(port)
            _terminate_process(process)
            if attempt < max_attempts:
                continue
            raise RuntimeError(last_error)

        print(f"Opening {url}")
        if not _open_url(url):
            print(f"Automatic browser launch failed. Open this URL manually: {url}")
        return

    if last_error:
        raise RuntimeError(last_error)


if __name__ == "__main__":
    main()
