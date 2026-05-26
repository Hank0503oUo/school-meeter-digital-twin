# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch

import panel as pn

from src.dashboard import create_dashboard, create_knowledge_workbench
from src.dashboard_modules.assistant_views import AssistantController
from src.dashboard_modules.map_views import MapViewController
from src.dashboard_modules.runtime import DashboardRuntime
from src.dashboard_modules import factory
from tests.dashboard._utils import build_dashboard_test_widgets, find_first


class TestDashboardLazyStartup(unittest.TestCase):
    def test_create_dashboard_returns_fast_list_template(self):
        template = create_dashboard()
        self.assertIsInstance(template, pn.template.FastListTemplate)

    def test_create_knowledge_workbench_returns_fast_list_template(self):
        template = create_knowledge_workbench()
        self.assertIsInstance(template, pn.template.FastListTemplate)

    def test_create_dashboard_does_not_eagerly_reload_campus(self):
        callbacks = []
        with (
            patch.object(factory.pn.state, "onload", side_effect=lambda callback: callbacks.append(callback)),
            patch.object(
                factory.DashboardRuntime,
                "reload_campus_state",
                side_effect=AssertionError("reload_campus_state should not run during create_dashboard"),
            ),
        ):
            template = factory.create_dashboard()

        self.assertIsInstance(template, pn.template.FastListTemplate)
        self.assertEqual(len(callbacks), 1)

    def test_create_dashboard_does_not_initialize_assistant_service(self):
        callbacks = []
        with (
            patch.object(factory.pn.state, "onload", side_effect=lambda callback: callbacks.append(callback)),
            patch(
                "src.demo_assistant.CampusAssistantService",
                side_effect=AssertionError("assistant service should be lazy"),
            ),
        ):
            template = factory.create_dashboard()

        self.assertIsInstance(template, pn.template.FastListTemplate)
        self.assertEqual(len(callbacks), 1)

    def test_dashboard_runtime_initializes_assistant_service_on_first_use(self):
        runtime = DashboardRuntime()

        with patch("src.demo_assistant.CampusAssistantService") as service_cls:
            service = runtime.assistant_service
            self.assertIs(service, service_cls.return_value)
            self.assertIs(runtime.assistant_service, service)

        service_cls.assert_called_once()

    def test_onload_callback_loads_campus_and_clears_spinner(self):
        callbacks = []

        def fake_load(self, campus_id: str) -> bool:
            self.active_campus_id = campus_id
            self.active_campus_name = "NTU"
            self.active_campus_ready = True
            self.campus_loaded = True
            self.loaded_campus_id = campus_id
            self.engine_mode = "Fallback"
            return True

        with (
            patch.object(factory.pn.state, "onload", side_effect=lambda callback: callbacks.append(callback)),
            patch.object(factory.DashboardRuntime, "load_campus", autospec=True, side_effect=fake_load) as load_campus,
            patch.object(factory.MapViewController, "refresh_campus_controls", autospec=True),
            patch.object(factory.AssistantController, "sync_nekaise_context", autospec=True),
            patch.object(factory, "trigger_dashboard_recompute"),
            patch.object(factory, "_run_on_next_tick", side_effect=lambda callback: callback()),
        ):
            template = factory.create_dashboard()
            spinner = find_first(template, pn.indicators.LoadingSpinner)
            self.assertIsNotNone(spinner)
            self.assertTrue(spinner.value)

            callbacks[0]()

            self.assertEqual(load_campus.call_count, 1)
            self.assertFalse(spinner.value)

    def test_assistant_controller_lazy_loads_workbench(self):
        widgets = build_dashboard_test_widgets()
        runtime = DashboardRuntime()

        with patch("src.dashboard_modules.assistant_views._load_nekaise_dashboard_class") as loader:
            workbench_cls = loader.return_value
            workbench_cls.return_value.build_embedded.return_value = pn.pane.Markdown("Knowledge ready")

            controller = AssistantController(runtime, widgets)
            controller.sync_nekaise_context()

            loader.assert_not_called()

            controller.ensure_workbench_loaded()

            loader.assert_called_once()
            workbench_cls.assert_called_once()
            workbench_cls.return_value.set_external_context.assert_called_once()

    def test_map_panel_renders_shell_map_before_campus_load_finishes(self):
        widgets = build_dashboard_test_widgets()
        runtime = DashboardRuntime()
        runtime.prepare_campus_shell("ntu")
        controller = MapViewController(runtime, widgets, lambda _days, _meter: {"dci": 0.0, "map_saturation": 1.0})

        panel = controller.map_panel(2020, "tier", "All", "all", 30, "ALL")

        self.assertIsNotNone(find_first(panel, pn.pane.DeckGL))


if __name__ == "__main__":
    unittest.main()
