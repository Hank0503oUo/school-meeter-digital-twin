# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path
from unittest.mock import call, patch

import open_browser_launcher as launcher


class TestOpenBrowserLauncher(unittest.TestCase):
    def test_load_config_uses_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "ENERGY_DEMO_PORT": "5012",
                "ENERGY_DEMO_FALLBACK_PORTS": "5013, 5014, invalid, 5013",
                "ENERGY_DEMO_WAIT_SECONDS": "120",
                "ENERGY_DEMO_START_RETRIES": "2",
            },
            clear=True,
        ):
            config = launcher._load_config()

        self.assertEqual(config.preferred_port, 5012)
        self.assertEqual(config.fallback_ports, (5013, 5014))
        self.assertEqual(config.wait_seconds, 120)
        self.assertEqual(config.start_retries, 2)

    def test_load_config_keeps_default_fallbacks_for_custom_preferred_port(self):
        with patch.dict(os.environ, {"ENERGY_DEMO_PORT": "5012"}, clear=True):
            config = launcher._load_config()

        self.assertEqual(config.preferred_port, 5012)
        self.assertEqual(config.fallback_ports, (5006, 5008, 5009, 5010, 5011))

    def test_classify_demo_html(self):
        html = "<html><title>NTU 校園能源數位分身 ｜ PI-VD</title></html>"
        self.assertEqual(launcher._classify_app_html(html), "demo")

    def test_classify_workbench_html(self):
        html = "<html><title>Building Energy Knowledge Workbench</title></html>"
        self.assertEqual(launcher._classify_app_html(html), "workbench")

    def test_choose_target_port_reuses_demo_when_present(self):
        probe_map = {5006: "demo", 5008: None}

        def probe(port):
            return probe_map.get(port)

        def port_in_use(_port):
            return False

        port, should_start = launcher._choose_target_port(
            preferred_port=5006,
            fallback_ports=(5008,),
            probe_app_fn=probe,
            port_in_use_fn=port_in_use,
        )
        self.assertEqual(port, 5006)
        self.assertFalse(should_start)

    def test_choose_target_port_skips_wrong_existing_app(self):
        probe_map = {5006: "workbench", 5008: None}
        busy_map = {5006: True, 5008: False}

        def probe(port):
            return probe_map.get(port)

        def port_in_use(port):
            return busy_map.get(port, False)

        port, should_start = launcher._choose_target_port(
            preferred_port=5006,
            fallback_ports=(5008,),
            probe_app_fn=probe,
            port_in_use_fn=port_in_use,
            reset_port_fn=lambda _port: False,
        )
        self.assertEqual(port, 5008)
        self.assertTrue(should_start)

    def test_choose_target_port_resets_stale_demo_server_on_preferred_port(self):
        probe_map = {5006: "workbench", 5008: None}
        busy_map = {5006: True, 5008: False}

        def probe(port):
            return probe_map.get(port)

        def port_in_use(port):
            return busy_map.get(port, False)

        def reset_port(port):
            self.assertEqual(port, 5006)
            busy_map[5006] = False
            probe_map[5006] = None
            return True

        port, should_start = launcher._choose_target_port(
            preferred_port=5006,
            fallback_ports=(5008,),
            probe_app_fn=probe,
            port_in_use_fn=port_in_use,
            reset_port_fn=reset_port,
        )
        self.assertEqual(port, 5006)
        self.assertTrue(should_start)

    def test_choose_target_port_skips_excluded_failed_ports(self):
        probe_map = {5006: None, 5008: None}
        busy_map = {5006: False, 5008: False}

        port, should_start = launcher._choose_target_port(
            preferred_port=5006,
            fallback_ports=(5008,),
            excluded_ports={5006},
            probe_app_fn=lambda port: probe_map.get(port),
            port_in_use_fn=lambda port: busy_map.get(port, False),
        )

        self.assertEqual(port, 5008)
        self.assertTrue(should_start)

    def test_probe_http_ready_accepts_any_http_response(self):
        self.assertTrue(
            launcher._probe_http_ready(
                5006,
                fetch_text_fn=lambda _url, timeout=2.0: "<html>generic panel shell</html>",
            )
        )

    def test_probe_http_ready_rejects_missing_response(self):
        self.assertFalse(
            launcher._probe_http_ready(
                5006,
                fetch_text_fn=lambda _url, timeout=2.0: None,
            )
        )

    def test_wait_for_http_ready_accepts_slow_app_with_longer_timeout(self):
        clock = {"now": 0.0}
        http_attempts = {"count": 0}

        def time_fn():
            return clock["now"]

        def sleep_fn(seconds):
            clock["now"] += seconds

        def port_in_use(_port):
            return clock["now"] >= 1.0

        def probe_http_ready(_port, timeout=2.0):
            self.assertEqual(timeout, 8.0)
            http_attempts["count"] += 1
            return http_attempts["count"] >= 2

        ready, error = launcher._wait_for_http_ready(
            5006,
            timeout_seconds=20,
            http_probe_timeout=8.0,
            port_in_use_fn=port_in_use,
            probe_http_ready_fn=probe_http_ready,
            sleep_fn=sleep_fn,
            time_fn=time_fn,
        )

        self.assertTrue(ready)
        self.assertIsNone(error)
        self.assertGreaterEqual(clock["now"], 2.0)

    def test_wait_for_http_ready_reports_http_timeout_after_port_listens(self):
        clock = {"now": 0.0}

        def time_fn():
            return clock["now"]

        def sleep_fn(seconds):
            clock["now"] += seconds

        ready, error = launcher._wait_for_http_ready(
            5006,
            timeout_seconds=5,
            port_in_use_fn=lambda _port: True,
            probe_http_ready_fn=lambda _port, timeout=8.0: False,
            sleep_fn=sleep_fn,
            time_fn=time_fn,
        )

        self.assertFalse(ready)
        self.assertIn("/app", error)

    def test_wait_for_http_ready_reports_process_early_exit(self):
        class ExitedProcess:
            def poll(self):
                return 3

        ready, error = launcher._wait_for_http_ready(
            5006,
            timeout_seconds=5,
            process=ExitedProcess(),
            log_path=Path("panel_server_5006.log"),
            port_in_use_fn=lambda _port: False,
        )

        self.assertFalse(ready)
        self.assertIn("exit code 3", error)
        self.assertIn("panel_server_5006.log", error)

    def test_main_retries_on_next_port_after_start_failure(self):
        process_1 = object()
        process_2 = object()
        config = launcher.LauncherConfig(
            preferred_port=5006,
            fallback_ports=(5008,),
            wait_seconds=5,
            start_retries=1,
        )

        with (
            patch.object(launcher, "_load_config", return_value=config),
            patch.object(launcher, "_choose_target_port", side_effect=[(5006, True), (5008, True)]) as choose_port,
            patch.object(launcher, "apply_mcp_profile_menu_if_requested", return_value={}) as menu,
            patch.object(launcher, "run_startup_hooks", return_value={"ran_any": False}) as hooks,
            patch.object(launcher, "print_hook_summary"),
            patch.object(launcher, "_panel_log_path", side_effect=lambda port: Path(f"panel_{port}.log")),
            patch.object(launcher, "_start_demo_server", side_effect=[process_1, process_2]) as start_server,
            patch.object(launcher, "_wait_for_http_ready", side_effect=[(False, "first failure"), (True, None)]),
            patch.object(launcher, "_terminate_process") as terminate,
            patch.object(launcher, "_open_url", return_value=True) as open_url,
        ):
            launcher.main()

        self.assertEqual(choose_port.call_count, 2)
        start_server.assert_has_calls(
            [
                call(5006, log_path=Path("panel_5006.log")),
                call(5008, log_path=Path("panel_5008.log")),
            ]
        )
        terminate.assert_called_once_with(process_1)
        menu.assert_called_once()
        hooks.assert_called_once()
        open_url.assert_called_once_with("http://127.0.0.1:5008/app")


if __name__ == "__main__":
    unittest.main()
